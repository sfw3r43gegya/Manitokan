import copy, os

import torch

from components.episode_buffer import EpisodeBatch
from modules.critics.basic_distributed import BasicDistributedCritic
from utils.rl_utils import build_td_lambda_targets
import torch as th
import numpy as np
from torch.optim import RMSprop
from utils.rl_utils import LinearDecayScheduler


class PGLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger
        self.device = args.device
        self.spike = args.spike_variation
        self.max_steps = args.t_max
        self.anti_coop = -1 if args.anti_coop else 1
        self.hess_lr = float(args.hess_lr)
        self.td_res = int(args.td_resolution)

        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.episode = 0
        self.la_update = args.la_update
        self.obj1_lr = float(args.obj_1lr)

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.graph = args.create_hessian_graph
        self.naive=args.naive_learner

        self.critic = BasicDistributedCritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.lola = args.lola
        self.only_critics = args.only_critics

        self.agent_params = list(mac.parameters())
        self.critic_params = list(self.critic.parameters())
        self.params = self.agent_params + self.critic_params
        self.entropy_method = args.entropy_method
        self.policy_ent_coeff = args.policy_ent_coeff
        self.eps_limit = args.eps_limit
        self.h_lr = args.h_lr
        self.update_order=args.update_order
        self.kick_in = int(args.kick_in)
        self.his_disc=args.his_disc
        self.agent_optimiser = RMSprop(params=self.agent_params,
                                       lr=args.lr,
                                       alpha=args.optim_alpha,
                                       eps=args.optim_eps)  if not args.iid_agents else [RMSprop(params=x,lr=args.lr,
                                                                                                alpha=args.optim_alpha,
                                                                                                eps=args.optim_eps) for x in self.agent_params]

        self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.step=0
        self.Linear_Decay = LinearDecayScheduler(-args.h_lr,
                                                 -args.t_max * self.eps_limit)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int,  new_rewards=None, redist_steps=None, iter_over=None):
        # Get the relevant quantities
        self.step += 1
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = new_rewards.squeeze() if new_rewards is not None else batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask = mask.to(self.args.device)
        avail_actions = batch["avail_actions"][:, :-1]
        order = batch["order"]

        critic_mask = mask.clone()

        rewards= rewards.to(self.device)
        terminated = terminated.to(self.device)
        actions = actions.to(self.device)
        avail_actions = avail_actions.to(self.device)
        critic_mask = critic_mask.to(self.device)

        mask = mask.repeat(1, 1, self.n_agents)#.view(-1)


        mac_out = []
        ents = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t )
            if self.entropy_method == 'max':
                entropy = self.mac.action_selector.select_actions(agent_outs, avail_actions[:,t], t, ent=True)
                ents.append(entropy.detach())
            mac_out.append(agent_outs)

        mac_out = th.stack(mac_out, dim=1)

        if self.spike:
            rewards = self.spike_injection(rewards)

        # Mask out unavailable actions, renormalise (as in action selection)
        if self.la_update:
           # if self.step >= 4000:
             #   self.h_lr = -max(-self.Linear_Decay.step(), -0.0005)
            if self.lola:

                actions = actions[:, :-1]
                mac_out[avail_actions == 0] = 0




                mac_out = mac_out / mac_out.sum(dim=-1, keepdim=True) if self.args.env_args[
                                                                             "name"] != "starcraft" else mac_out
                mac_out[avail_actions == 0] = 0

                if self.entropy_method == 'max':
                    entropy = th.stack(ents, dim=1)
                    rewards -= self.policy_ent_coeff * entropy
                    ent = entropy.unsqueeze(3).clone()


                vals, critic_train_stats = self._train_critic(batch,
                                                              rewards, terminated, actions,
                                                              avail_actions,
                                                              critic_mask, bs, max_t, iter_over=order)

                # v = v*l
                pi = mac_out.to(self.args.device).squeeze()

                # mask = mask.reshape(pi_taken.shape)  # .view(-1
                pi_taken = th.gather(pi, dim=3, index=actions).squeeze()

                loss = self.loss_func(pi_taken,
                                      mask,
                                      vals,
                                      rewards,
                                      terminated,
                                      critic_mask,
                                      actions)

                p = (self.step) / (self.max_steps)

                p = 1 if p > 1 else p


                hess = self.hessian(self.loss_func, (pi_taken,
                                             mask,
                                             vals,
                                             rewards ,
                                             terminated,
                                             critic_mask,
                                                     actions))



                hess = hess[:, :, ]

                if self.naive:
                    hess[:, :, 0] = 0


                pg_loss = -(loss + 0.0005*loss[:,:,[1,0]]*hess).sum() / mask.sum()

                if type(self.agent_optimiser) is not list:
                    self.agent_optimiser.zero_grad()

                else:
                    for opt in self.agent_optimiser:
                        opt.zero_grad()

                pg_loss.backward()

                critic_train_stats["hessian_agent_1"] = (loss*hess).sum() / mask.sum()


            else:
                actions = actions[:, :-1]
                mac_out[avail_actions == 0] = 0
                r_i = rewards.unsqueeze(3).clone()
                r_i[r_i >= 1.0] = 0

                r_c = rewards.unsqueeze(3).clone()
                r_c[r_c < 1.0] = 0
                col = r_c.clone()
                r_c_index = r_c
                r_c = r_c.sum()/1.5
                mac_out = mac_out/mac_out.sum(dim=-1,keepdim=True) if self.args.env_args["name"] != "starcraft" else mac_out
                mac_out[avail_actions == 0] = 0
                l = mac_out.clone()
                r_mask = rewards.clone().sum()/(32*(0.5+0.5*self.n_agents)*self.n_agents)

                losses = []
                hessians = []

                if self.entropy_method == 'max':
                    entropy = th.stack(ents, dim=1)
                    rewards -= self.policy_ent_coeff * entropy
                    ent = entropy.unsqueeze(3).clone()
                    ent_i = ent.clone()
                    ent_c = ent.clone()
                    ent_i[r_i == 0.0] = 0
                    ent_c[col == 0.0] = 0
                    r_i -= self.policy_ent_coeff * ent_i
                    col -= self.policy_ent_coeff * ent_c
                    col = col.clone()

                vals, critic_train_stats = self._train_critic(batch,
                                                              rewards, terminated, actions,
                                                              avail_actions,
                                                              critic_mask, bs, max_t, iter_over=order)



                # v = v*l
                pi = mac_out.to(self.args.device).squeeze()



                #mask = mask.reshape(pi_taken.shape)  # .view(-1
                pi_taken = th.gather(pi,
                                     dim=3,
                                     index=actions).squeeze()

                p = (self.step) / (self.max_steps+self.kick_in)
                p = 1 if p > 1 else p
                #if self.his_disc:
                 #   rewards[rewards == 0.5] = (1-p)*rewards[rewards == 0.5]

                loss = self.loss_func(pi_taken,
                                             mask,
                                             vals,
                                             rewards ,
                                             terminated,
                                             critic_mask, actions)


                if self.step >= self.kick_in:


                    l = th.gather(l, dim=3, index=actions).squeeze(1)
                    v = (vals.clone() - r_i * l)
                    prob_vec = 1 / th.log(l)  # vectorize hessian

                    prob_vec[r_c_index == 0 ] = 0


                    hess = self.hessian(self.loss_func, (l.squeeze(),
                                                              mask,
                                                              v.squeeze(),
                                                              col.squeeze(),
                                                              terminated,
                                                              critic_mask, actions))

                    hess = torch.einsum('ijlkk,ijlo->ijl', hess, prob_vec)

                    if self.update_order:
                        hess = hess[:,:,[1,0]]

                    if self.naive:
                        hess[:,:,0] = 0

                    if self.his_disc :
                        l_i = l.clone()
                        vi =   (vals.clone() - r_c * l_i)
                        prob_vec_i = 1/l_i

                        prob_vec_i[r_i == 0] = 0
                        ind = r_i.clone()

                        hess_i = self.hessian(self.loss_func, (l_i.squeeze(),
                                                             mask,
                                                             vi.squeeze(),
                                                             ind.squeeze(),
                                                             terminated,
                                                             critic_mask, actions))

                        hess_i = -1*self.anti_coop * torch.einsum('ijlkk,ijlo->ijl', hess_i,  prob_vec_i)




                else:
                    p = 0
                    hess = torch.Tensor([0]).to(self.args.device)




            if self.his_disc:
                pg_loss = -(loss +   hess + hess_i).sum() / mask.sum()

            else:
                pg_loss = -( loss*self.obj1_lr +  hess).sum() / mask.sum()

            if type(self.agent_optimiser) is not list:
                self.agent_optimiser.zero_grad()

            else:
                for opt in self.agent_optimiser:
                    opt.zero_grad()

            pg_loss.backward()



            critic_train_stats["hessian_agent_1"] =   self.anti_coop*hess.sum()/mask.sum()
          #  critic_train_stats["r_mask"] = r_mask
            if self.his_disc:
               critic_train_stats["hessian_individual"] = -1 * self.anti_coop * hess_i.sum() / mask.sum()
          #  critic_train_stats["norm_hessian_agent_1"] = h1_norm.item()

      #      critic_train_stats["Hessian_difference"] = abs((0.0005*(p))*h1_loss.sum()/mask.sum()- (0.0005*(p))*h2_loss.sum()/mask.sum())

       #     critic_train_stats["hessian_agent_2"] = (0.0005*(p))*h2_loss.sum()/mask.sum() if  self.naive else (0.0005*(p))*h2_loss.sum()/mask.sum()
           # critic_train_stats["norm_hessian_agent_2"] = h2_norm.item()




        else:
            if self.entropy_method == 'max':
                entropy = th.stack(ents, dim=1)
                rewards -= self.policy_ent_coeff * entropy
            mask = mask.view(-1)
            actions = actions[:, :-1]
            mac_out[avail_actions == 0] = 0

            mac_out = mac_out / mac_out.sum(dim=-1, keepdim=True) if self.args.env_args["name"] != "starcraft" else mac_out
            mac_out[avail_actions == 0] = 0
            pi = mac_out.view(-1, self.n_actions)
            pi = pi.to(self.args.device)

            vals, critic_train_stats = self._train_critic(batch,
                                                          rewards, terminated, actions,
                                                          avail_actions,
                                                          critic_mask, bs, max_t, iter_over=order)

            baseline = vals.squeeze(-1).to(self.args.device)

            # Calculate policy grad with mask
            # th.cat([th.zeros_like(vals), th.zeros_like(vals)[:, -1:]], 1).squeeze(3)
            ret = build_td_lambda_targets(rewards,
                                          terminated,
                                          critic_mask,
                                          th.zeros_like(th.cat([rewards, rewards[:, -1:]], 1), device=self.args.device),
                                          self.n_agents,
                                          self.args.gamma,
                                    td_lambda=1.0)
            pi_taken = th.gather(pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
            pi_taken[mask == 0] = 1.0

            log_pi_taken = th.log(pi_taken).to(self.args.device)
            advantages = (ret - baseline).reshape(-1).detach()
            pg_loss = - ((advantages * log_pi_taken) * mask).sum() / mask.sum()
            if type(self.agent_optimiser) is not list:
                self.agent_optimiser.zero_grad()

            else:
                for opt in self.agent_optimiser:
                    opt.zero_grad()

            pg_loss.backward()


        # Optimise agents

        if type(self.agent_optimiser) is not list:
            self.agent_optimiser.step()

        else:
            for opt in self.agent_optimiser:
                opt.step()

        if (self.critic_training_steps - self.last_target_update_step) / int(self.args.target_update_interval) >= 1.0:
            self._update_targets()
            self.last_target_update_step = self.critic_training_steps

        if self.entropy_method == 'max':
            critic_train_stats["average_entropy"] = th.mean(entropy.float()).item()
        critic_train_stats["actor_loss"] = pg_loss.item()


        return critic_train_stats

    def _train_critic(self, batch, rewards, terminated, actions, avail_actions, mask, bs, max_t, iter_over= None):
        # Optimise critic

        target_vals = self.target_critic(batch, agent_order=iter_over)[:, :]

        # Calculate td-lambda targets
        targets = build_td_lambda_targets(rewards, terminated, mask, target_vals.squeeze(3), self.n_agents, self.args.gamma, self.args.td_lambda)
        targets = targets.to(self.args.device)
        vals = th.zeros_like(target_vals, device=self.args.device)[:, :-1]

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "v_taken_mean": [],
        }
        if iter_over is None:
            iter_over = range(self.n_agents)

        if self.td_res == 1:

            for t in reversed(range(rewards.size(1))):
                mask_t = mask[:, t]
                mask_t = mask_t.expand(-1,  self.n_agents).to(self.args.device).squeeze()
                if mask_t.sum() == 0:
                    continue

                v_t = self.critic(batch, t,iter_over)
                vals[:, t] = v_t.view(bs, self.n_agents, 1)
                v_taken = v_t.squeeze(3).squeeze(1)
                targets_t = targets[:, t]
                td_error = (v_taken - targets_t.detach())

                # 0-out the targets that came from padded data
                masked_td_error = td_error * mask_t

                # Normal L2 loss, take mean over actual data
                loss = (masked_td_error ** 2).sum() / mask_t.sum()
                self.critic_optimiser.zero_grad()
                loss.backward()
                grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
                self.critic_optimiser.step()
                self.critic_training_steps += 1

                running_log["critic_loss"].append(loss)
                running_log["critic_grad_norm"].append(grad_norm)
                mask_elems = mask_t.sum().item()
                running_log["td_error_abs"].append((masked_td_error.abs().sum() / mask_elems))
                running_log["v_taken_mean"].append((v_taken * mask_t).sum() / mask_elems)
                running_log["target_mean"].append((targets_t * mask_t).sum() / mask_elems)
        else:
            for t in reversed(range(0, rewards.size(1), self.td_res)):
                k = slice( t,t+self.td_res)
                mask_t = mask[:, t:t+self.td_res,:]#.expand(-1, self.n_agents).to(self.args.device)
                mask_t =  mask_t.expand(-1,-1, self.n_agents).to(self.args.device).squeeze()
                if mask_t.sum() == 0:
                    continue

                v_t = self.critic(batch, k,  iter_over,)

                vals[:, k] = v_t.view(bs,self.td_res, self.n_agents,1)
                v_taken = v_t.squeeze(3).squeeze(1) if self.td_res == 1 else v_t.squeeze(3)
                targets_t = targets[:, k].squeeze()
                td_error = (v_taken - targets_t.detach())

                # 0-out the targets that came from padded data
                masked_td_error = td_error * mask_t

                # Normal L2 loss, take mean over actual data
                loss = (masked_td_error ** 2).sum() / mask_t.sum()
                if not self.only_critics:
                    self.critic_optimiser.zero_grad()
                    loss.backward()
                    grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
                    self.critic_optimiser.step()
                    running_log["critic_grad_norm"].append(grad_norm)
                self.critic_training_steps += 1

                running_log["critic_loss"].append(loss)

                mask_elems = mask_t.sum().item()
                running_log["td_error_abs"].append((masked_td_error.abs().sum() / mask_elems))
                running_log["v_taken_mean"].append((v_taken * mask_t).sum() / mask_elems)
                running_log["target_mean"].append((targets_t * mask_t).sum() / mask_elems)

        for key in running_log:
            if type(running_log[key]) is list:
                running_log[key] = torch.stack(running_log[key])
            running_log[key] = th.mean(running_log[key]).item()

        return vals, running_log


    def loss_func(self, pi_taken, mask,  values,  rewards, terminated, critic_mask, actions):


        baseline =  values.squeeze(-1).to(self.args.device)

        # Calculate policy grad with mask
        # th.cat([th.zeros_like(vals), th.zeros_like(vals)[:, -1:]], 1).squeeze(3)
        ret = build_td_lambda_targets(rewards,
                                      terminated,
                                      critic_mask,
                                      th.zeros_like(th.cat([rewards, rewards[:, -1:]], 1), device=self.args.device),
                                      self.n_agents,
                                      self.args.gamma,
                                td_lambda=1.0)


        pi_taken[mask == 0] = 1.0

        log_pi_taken = th.log(pi_taken).to(self.args.device)
        advantages = (ret - baseline).detach()
        pg_loss =  ((advantages * log_pi_taken) * mask)

        return pg_loss

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
#        self.logger.console_logger.info("Updated target network")

    def spike_injection(self, rewards):
        eps = 15
        decay = 1-(self.episode/10000)
        new_rewards = rewards.clone()
        self.episode +=1
        for batch in range(rewards.shape[0]):
            for agent in range(rewards.shape[2]):
                first_reward = False
                second_reward = False
                for t in range(rewards.shape[1]):
                    if rewards[batch, t, agent].item() == 0.5 and not first_reward:
                        first_reward = True
                        mean_idx = t

                        for i in range(max(mean_idx-eps,0),mean_idx+1):
                            sample = np.random.normal(rewards[batch, t, agent],1, 1)[0]
                            new_rewards[batch, i, agent] += decay*th.abs(sample-rewards[batch, t, agent])/(max(mean_idx,1) -i +1)*self.policy_ent_coeff

                        for i in range(min(299,mean_idx+1),min(mean_idx+eps,299)):
                            sample = np.random.normal(rewards[batch, t, agent], 1, 1)[0]
                            new_rewards[batch, i, agent] += decay*th.abs(sample-rewards[batch, t, agent])/(i+eps-mean_idx+1)*self.policy_ent_coeff

                    if rewards[batch, t, agent].item() >= 1.5 and not second_reward:
                        second_reward = True
                        mean_idx = t

                        for i in range(max(mean_idx-eps,0),mean_idx+1):
                            sample = np.random.normal(rewards[batch, t, agent], eps, 1)[0]
                            new_rewards[batch, i, agent] += decay*th.abs(sample-rewards[batch, t, agent])/(max(mean_idx,1) -i +1)*self.policy_ent_coeff

        return new_rewards

    def hessian(self, loss_func, memories, eps=1e-6, ):
        probs, masks, vals_adjusted, rews, dones, masks_critic, actions = memories

        h = torch.zeros((32, 300, self.n_agents, 2, 2)).to(self.device) # shape is trajectory
        index_0 = [0 for _ in range(2)]
        for i in range(2):
            index_0[i] = 1
            index_1 = [0 for _ in range(2)]

            for j in range(2):
                index_1[j] = 1

                gradient0 = loss_func(probs -eps*.5*index_0[0] , masks, vals_adjusted -eps*.5*index_0[1], rews , dones, masks_critic, actions) # add -eps*.5 to vals and rewards and return a grad in batch, trajectory, val
                gradient1 = loss_func(probs +eps*.5* index_1[0], masks, vals_adjusted +eps*.5*index_1[1], rews, dones, masks_critic, actions) # add -eps*.5 to vals and rewards and return a grad in batch, trajectory, val


                index_1[j] = 0

                h[:,:,:,i,j] = (gradient1 - gradient0)/(eps)

            index_0[i] = 0

        return h


    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):

        if not self.only_critics:
            self.mac.save_models(path)
            th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))

        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.target_critic.state_dict(), "{}/target_critic.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path, path2=None):
        if os.path.isfile(path):
            if not self.only_critics:
                self.mac.load_models(path)
                self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))

            self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
            self.target_critic.load_state_dict(th.load("{}/target_critic.th".format(path), map_location=lambda storage, loc: storage))
            c_opt_dic = th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage)
            states = c_opt_dic["state"]

            if int(self.args.load_agentn) < self.n_agents:
                state_len =len(states)

                for i in range(state_len):
                    c_opt_dic["state"][state_len+i] = states[i]
                    c_opt_dic['param_groups'][0]['params'].append(state_len+i)

            self.critic_optimiser.load_state_dict(c_opt_dic)
