import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.coma import COMACritic
from utils.rl_utils import build_td_lambda_targets
import torch as th
import numpy as np
from torch.optim import RMSprop


class COMALearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.la_update = args.la_update
        self.step = 0
        self.device = args.device

        self.critic = COMACritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.entropy_method = args.entropy_method
        self.policy_ent_coeff = args.policy_ent_coeff

        self.agent_params = list(mac.parameters())
        self.critic_params = list(self.critic.parameters())
        self.params = self.agent_params + self.critic_params

        self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, new_rewards=None, redist_steps=None, buffer = None):
        # Get the relevant quantities
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = new_rewards if new_rewards is not None else batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]

        critic_mask = mask.clone()

        mask = mask.repeat(1, 1, self.n_agents)#.view(-1)

        q_vals, critic_train_stats = self._train_critic(batch, rewards, terminated, actions, avail_actions,
                                                        critic_mask, bs, max_t)

        actions = actions[:,:-1]


        mac_out = []
        ents=[]
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            if self.entropy_method == 'max':
                entropy = self.mac.action_selector.select_actions(agent_outs, avail_actions[:,t], t, ent=True)
                ents.append(entropy.detach())

        mac_out = th.stack(mac_out, dim=1)  # Concat over time


        if self.entropy_method == 'max':
            entropy = th.stack(ents, dim=1)
            rewards -= self.policy_ent_coeff *entropy
            critic_train_stats["average_entropy"] = entropy.detach().cpu().numpy()


        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)  if self.args.env_args["name"] != "starcraft" else mac_out
        mac_out[avail_actions == 0] = 0


        if self.la_update:
            r_i = rewards.unsqueeze(3).clone()
            r_i[r_i >= 1.0] = 0

            r_c = rewards.unsqueeze(3).clone()
            r_c[r_c < 1] = 0
            coc = r_c.clone()
            r_c_index = r_c
            r_c = r_c.sum() / 1.5
            l = mac_out.clone()
            r_mask = rewards.clone().sum()/(32*(0.5+0.5*self.n_agents)*self.n_agents)

            loss = self.loss_func( mac_out, mask,  q_vals, actions )

            if True:
                p = 1#r_c / (32)
               # l = th.gather(l, dim=3, index=actions).squeeze(1)
                v = (q_vals.clone() - r_i * l)
                prob_vec = 1 / l  # vectorize hessian
                prob_vec = th.gather(prob_vec, dim=3, index=actions).squeeze(1)
                prob_vec[r_c_index == 0] = 0


                hess = self.hessian(self.loss_func, ( l, mask,  v, actions ))

                hess = -th.einsum('ijlkk,ijlo->ijl', hess, prob_vec)
            else:
                p = 0
                hess = torch.Tensor([0]).to(self.args.device)


            self.agent_optimiser.zero_grad()
            coma_loss = -(loss+hess).sum()/mask.sum()
            coma_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
            self.agent_optimiser.step()

            for key in critic_train_stats:

                    critic_train_stats[key] = np.mean(critic_train_stats[key]).item()

            critic_train_stats["hessian_agent_1"] = (p) * hess.sum() / mask.sum()






        else:
            # Calculated baseline
            q_vals[avail_actions == 0] = 1e-10
            q_vals = q_vals.reshape(-1, self.n_actions)


            pi = mac_out.view(-1, self.n_actions)
            baseline = (pi * q_vals).sum(-1).detach()
            mask = mask.view(-1)
            # Calculate policy grad with mask
            q_taken = th.gather(q_vals, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
            pi_taken = th.gather(pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
            pi_taken[mask == 0] = 1.0
            log_pi_taken = th.log(pi_taken)

            advantages = (q_taken - baseline).detach()
            coma_loss = - ((advantages * log_pi_taken) * mask).sum() / mask.sum()


        # Optimise agents
            self.agent_optimiser.zero_grad()
            coma_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
            self.agent_optimiser.step()

            for key in critic_train_stats:

                    critic_train_stats[key] = np.mean(critic_train_stats[key]).item()

        if (self.critic_training_steps - self.last_target_update_step) / int(self.args.target_update_interval) >= 1.0:
            self._update_targets()
            self.last_target_update_step = self.critic_training_steps


        critic_train_stats["base_line"] = -baseline.sum()/mask.sum()
        critic_train_stats["actor_loss"] = coma_loss.item()

        return critic_train_stats

    def _train_critic(self, batch, rewards, terminated, actions, avail_actions, mask, bs, max_t):
        # Optimise critic
        target_q_vals = self.target_critic(batch)[:, :]
        targets_taken = th.gather(target_q_vals, dim=3, index=actions).squeeze(3)

        # Calculate td-lambda targets
        targets = build_td_lambda_targets(rewards, terminated, mask, targets_taken, self.n_agents, self.args.gamma, self.args.td_lambda)

        q_vals = th.zeros_like(target_q_vals)[:, :-1]

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        for t in reversed(range(rewards.size(1))):
            mask_t = mask[:, t].expand(-1, self.n_agents)
            if mask_t.sum() == 0:
                continue

            q_t = self.critic(batch, t)

            q_vals[:, t] = q_t.view(bs, self.n_agents, self.n_actions)
            q_taken = th.gather(q_t, dim=3, index=actions[:, t:t+1]).squeeze(3).squeeze(1)
            targets_t = targets[:, t]

            td_error = (q_taken - targets_t.detach())

            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask_t

            # Normal L2 loss, take mean over actual data
            loss = (masked_td_error ** 2).sum() / mask_t.sum()

            self.critic_optimiser.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
            self.critic_optimiser.step()
            self.critic_training_steps += 1

            running_log["critic_loss"].append(loss.item())
            running_log["critic_grad_norm"].append(grad_norm)
            mask_elems = mask_t.sum().item()
            running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
            running_log["q_taken_mean"].append((q_taken * mask_t).sum().item() / mask_elems)
            running_log["target_mean"].append((targets_t * mask_t).sum().item() / mask_elems)



        return q_vals, running_log

    def loss_func(self, mac_out, mask,  q_vals, actions  ):


        q_vals = q_vals.squeeze()#.reshape(-1, self.n_actions)
        pi = mac_out#.view(-1, self.n_actions)
        baseline = (pi * q_vals).sum(-1).detach()


        # Calculate policy grad with mask
        q_taken = th.gather(q_vals, dim=3, index=actions).squeeze()
        pi_taken = th.gather(pi, dim=3, index=actions).squeeze()
        pi_taken[mask == 0] = 1.0
        log_pi_taken = th.log(pi_taken)


        advantages = (q_taken - baseline).detach()
        coma_loss =  ((advantages * log_pi_taken) * mask)

        return coma_loss

    def hessian(self, loss_func, memories, eps=1e-6, ):
        probs, mask, q_vals, actions = memories

        h = th.zeros((32, 300, self.n_agents, 2, 2)).to(self.device) # shape is trajectory
        index_0 = [0 for _ in range(2)]

        for i in range(2):
            index_0[i] = 1
            index_1 = [0 for _ in range(2)]

            for j in range(2):
                index_1[j] = 1

                gradient0 = loss_func(probs -eps*.5*index_0[0] , mask, q_vals -eps*.5*index_0[1], actions) # add -eps*.5 to vals and rewards and return a grad in batch, trajectory, val
                gradient1 = loss_func(probs +eps*.5* index_1[0], mask, q_vals +eps*.5*index_1[1], actions) # add -eps*.5 to vals and rewards and return a grad in batch, trajectory, val


                index_1[j] = 0

                h[:,:,:,i,j] = (gradient1 - gradient0)/(eps)

            index_0[i] = 0



        return h

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
#        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
