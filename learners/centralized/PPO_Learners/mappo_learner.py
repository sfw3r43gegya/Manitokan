import math
import torch.nn
import torch as th
import numpy as np
from .PPO_Policies.valuenorm import ValueNorm
from .PPO_Policies.MappoPolicy import R_MAPPOPolicy


class MAPPOLearner:
    def __init__(self,  logger, args,  policy =R_MAPPOPolicy, mixer = None ):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions

        self.logger = logger
        self.learning_rate = args.lr

        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.max_grad_norm = args.use_max_grad_norm
        self.shared = args.share_buffer
        self.entropy_method = args.entropy_method
        self._use_softplus_entropy = args.use_softplus_entropy
        self._policy_ent_coeff = args.policy_ent_coeff
        self.la_update = args.la_update
        self.last_action_obs = args.last_action_obs


        self.minibatch_size = args.batch_size // args.batch_size
        self.num_mini_batch = int(args.num_mini_batch)
        self.device = torch.device(args.device)


        self.entropy_coef = args.entropy_coef

        if args.use_centralized_V and type(args.obs_shape) == int :
            s_obs = args.obs_shape * self.n_agents
        else:
            s_obs = args.obs_shape




        if self.shared:
            self.policy = policy(args,
                                 args.obs_shape,
                                 s_obs,
                                 self.n_actions,
                                 device=self.device)
        else:
            self.policy = [policy(args,
                                  args.obs_shape,
                                  s_obs,
                                  self.n_actions,
                                  device=self.device) for _ in range(self.n_agents)]


        self.clip_param = args.clip_param
        self.ppo_epoch = int(args.ppo_epoch)
        self.data_chunk_length = args.data_chunk_length


        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self.value_loss_coef = args.value_loss_coef
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_popart = args.use_popart
        self._use_huber_loss = args.use_huber_loss

        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        assert (self._use_popart and self._use_valuenorm) == False, (
            "self._use_popart and self._use_valuenorm can not be set True simultaneously")

        if self._use_popart and self.shared:
            self.value_normalizer = self.policy.critic.v_out

        elif self._use_popart:

            self.value_normalizer = [self.policy[i].critic.v_out for i in range(self.n_agents)] if not args.one_actor else  self.policy.critic.v_out

        elif self._use_valuenorm and self.shared:
            self.value_normalizer = ValueNorm(1).to(self.policy.device)

        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1).to(self.policy[0].device)
        else:
            self.value_normalizer = None



        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        if mixer is not None:
            self.mixer = mixer


        else:

            self.mixer = None



    def train(self, new_rewards=None, redist_steps=None, buffer = None, update_actor = True, indices=None):
        # Get the relevant quantities

        if self.shared:
            adv_returns = buffer.returns[:-1]



            if   self._use_valuenorm:
                advantages = adv_returns - self.value_normalizer.denormalize(buffer.value_preds[:-1])

            else:
                advantages =  adv_returns  - buffer.value_preds[:-1]

            advantages_copy = advantages.copy()
            advantages_copy[buffer.masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)


            train_info = {}

            train_info['value_loss'] = 0
            train_info['policy_loss'] = 0
            train_info['dist_entropy'] = 0
            train_info['actor_grad_norm'] = 0
            train_info['critic_grad_norm'] = 0
            train_info['ratio'] = 0

            for _ in range(self.ppo_epoch):


                    if self._use_recurrent_policy:
                        data_generator = buffer.recurrent_generator(advantages,
                                                                    self.num_mini_batch,
                                                                    self.data_chunk_length)

                    elif self._use_naive_recurrent:
                        data_generator = buffer.naive_recurrent_generator(advantages,
                                                                          self.num_mini_batch)

                    else:
                        data_generator = buffer.feed_forward_generator(advantages,
                                                                       self.num_mini_batch)



                    for sample in data_generator:
                        #check for new reward and step dsitrbution

                        value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                            = self._ppo_update(sample, update_actor, indices)

                        train_info['value_loss'] += value_loss.item()
                        train_info['policy_loss'] += policy_loss.item()
                        train_info['dist_entropy'] += dist_entropy.mean().item()
                        train_info['actor_grad_norm'] += actor_grad_norm.mean().item()
                        train_info['critic_grad_norm'] += critic_grad_norm.item()
                        train_info['ratio'] += imp_weights.mean().item()

        else:
            advantages = []
            train_info = {}
            for i in range(self.n_agents):

                if self._use_valuenorm:
                    advantages.append(buffer[i].returns[:-1] - self.value_normalizer.denormalize(buffer[i].value_preds[:-1]))

                else:
                    advantages.append(buffer[i].returns[:-1] - buffer[i].value_preds[:-1])

                advantages_copy = advantages[i].copy()
                advantages_copy[buffer[i].masks[:-1] == 0.0] = np.nan
                mean_advantages = np.nanmean(advantages_copy)
                std_advantages = np.nanstd(advantages_copy)
                advantages[i] = (advantages[i] - mean_advantages) / (std_advantages + 1e-5)



                train_info['agent '+ str(i+1) +' value_loss'] = 0
                train_info['agent '+ str(i+1) +' policy_loss'] = 0
                train_info['agent '+ str(i+1) +' dist_entropy'] = 0
                train_info['agent '+ str(i+1) +' actor_grad_norm'] = 0
                train_info['agent '+ str(i+1) +' critic_grad_norm'] = 0
                train_info['agent '+ str(i+1) +' ratio'] = 0

            for _ in range(self.ppo_epoch):
                data_generators = []

                episode_length, n_rollout_threads = buffer[0].rewards.shape[0:2]
                batch_size = n_rollout_threads * episode_length
                data_chunks = batch_size // self.data_chunk_length
                rand = th.randperm(data_chunks).cpu().numpy() if self.mixer is not None else None

                for i in range(self.n_agents):

                    if self._use_recurrent_policy:
                        data_generators.append(buffer[i].recurrent_generator(advantages[i],
                                                                             self.num_mini_batch,
                                                                             self.data_chunk_length,
                                                                             rand= rand))

                    elif self._use_naive_recurrent:
                        data_generators.append(buffer[i].naive_recurrent_generator(advantages[i],
                                                                                   self.num_mini_batch))

                    else:
                        data_generators.append(buffer[i].feed_forward_generator(advantages[i],
                                                                                self.num_mini_batch))



                i = 0
                for sample1, sample2 in zip(data_generators[0], data_generators[1]):
                    # check for new reward and step dsitrbution
                    i +=1

                    value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weight \
                        = self._ppo_update([sample1, sample2], update_actor, indices)


                    for j in range(self.n_agents):
                        train_info['agent ' + str(j+1) + ' value_loss'] += value_loss[j].item()
                        train_info['agent ' + str(j+1) + ' policy_loss'] += policy_loss[j].item()
                        train_info['agent ' + str(j+1) + ' dist_entropy'] += dist_entropy[j].mean().item()
                        train_info['agent ' + str(j+1) + ' actor_grad_norm'] += actor_grad_norm[j]
                        train_info['agent ' + str(j+1) + ' critic_grad_norm'] += critic_grad_norm[j]
                        train_info['agent ' + str(j+1) + ' ratio'] += imp_weight[j].mean()


        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def _ppo_update(self, sample, update_actor=True, indices=None):
        # Optimise critic
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """


        if self.shared:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch = sample

            # Reshape to do in a single forward pass for all steps

            if self.last_action_obs:
                new_obs = share_obs_batch.copy()
                share_obs_batch = share_obs_batch[:,:,0:self.args.obs_shape*2]
            else:
                new_obs = None

            values, action_log_probs, dist_entropy  = self.policy.evaluate_actions(cent_obs=share_obs_batch,
                                                                                      obs=obs_batch,
                                                                                      rnn_states_actor=rnn_states_batch,
                                                                                     rnn_states_critic= rnn_states_critic_batch,
                                                                                       action=actions_batch,
                                                                                      masks=masks_batch,
                                                                                      available_actions=available_actions_batch,
                                                                                      new_obs=new_obs) # cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                                # available_actions=None, active_masks=None
                # actor update
            if self.la_update:
                dist_entropy = dist_entropy*indices
                action_log_probs_collective = action_log_probs*indices
                old_action_log_probs_collective = old_action_log_probs_batch*indices

            imp_weights = th.exp(action_log_probs - torch.from_numpy(old_action_log_probs_batch).reshape(action_log_probs.shape).to(self.device))

            surr1 = imp_weights * torch.from_numpy(adv_targ).reshape(action_log_probs.shape).to(self.device)
            surr2 = th.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) *  torch.from_numpy(adv_targ).reshape(action_log_probs.shape).to(self.device)

            if self._use_policy_active_masks:
                    policy_action_loss = (-th.sum(th.min(surr1, surr2),
                                                     dim=-1,
                                                     keepdim=True) * torch.from_numpy(active_masks_batch).reshape(action_log_probs.shape).to(self.device)).sum() / torch.from_numpy(active_masks_batch).to(self.device).sum()
            else:
                    policy_action_loss = -th.sum(th.min(surr1, surr2), dim=-1, keepdim=True).mean()

            policy_loss = policy_action_loss


            if type(self.policy.actor_optimizer) is list:
                for i in range(self.n_agents):
                    self.policy.actor_optimizer[i].zero_grad()
            else:
                self.policy.actor_optimizer.zero_grad()

            if update_actor and self.entropy_method == "kl":
                        (policy_loss - dist_entropy.mean() * self.entropy_coef).backward()
            elif update_actor and self.entropy_method == "max":
                (policy_loss).backward()






            if self._use_max_grad_norm:

                actor_grad_norm = []
                if type(self.policy.actor_optimizer) is list:
                    for i in range(self.n_agents):
                            actor_grads = th.nn.utils.clip_grad_norm_(self.policy.actor[i].parameters(), self.max_grad_norm)
                            actor_grad_norm.append(actor_grads)

                    actor_grad_norm = th.stack(actor_grad_norm)

                else:

                    actor_grads = th.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
                    actor_grad_norm.append(actor_grads)

                    actor_grad_norm = th.stack(actor_grad_norm)

            else:

                actor_grad_norm = []
                if type(self.policy.actor_optimizer) is list:
                    for i in range(self.n_agents):
                        actor_grads = th.nn.utils.clip_grad_norm_(self.policy.actor[i].parameters())
                        actor_grad_norm.append(actor_grads)

                    actor_grad_norm = th.stack(actor_grad_norm)

                else:

                    actor_grads = th.nn.utils.clip_grad_norm_(self.policy.actor.parameters())
                    actor_grad_norm.append(actor_grads)

                    actor_grad_norm = th.stack(actor_grad_norm)

            if type(self.policy.actor_optimizer) is list:
                for i in range(self.n_agents):
                    self.policy.actor_optimizer[i].step()
            else:
                self.policy.actor_optimizer.step()



            # critic update

            if not torch.is_tensor(value_preds_batch):
                value_preds_batch = torch.from_numpy(value_preds_batch).to(self.device)

            value_loss = self.cal_value_loss(values,
                                             value_preds_batch.reshape(values.shape[0], -1),
                                             return_batch.reshape(values.shape[0], -1),
                                             active_masks_batch.reshape(values.shape[0], -1))

            self.policy.critic_optimizer.zero_grad()

            (value_loss * self.value_loss_coef).backward()

            if self._use_max_grad_norm:
                critic_grad_norm = th.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)

            else:
                critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

            self.policy.critic_optimizer.step()



            return value_loss.cpu().detach(), critic_grad_norm.cpu().detach(), policy_loss.cpu().detach(), dist_entropy.cpu().detach(), actor_grad_norm.cpu().detach(), imp_weights.cpu().detach()

        else:
            mix_vals = []
            value_losses = []
            critic_grad_norms = []
            policy_losses = []
            dist_entropies = []
            actor_grad_norms = []
            imp_weight = []

            preds = []
            rets = []
            a_masks = []
            acts = []
            s_obs = []
            for i in range(self.n_agents):

                (share_obs_batch,
                 obs_batch,
                 rnn_states_batch,
                 rnn_states_critic_batch,
                 actions_batch,
                 value_preds_batch,
                 return_batch,
                 masks_batch,
                 active_masks_batch,
                 old_action_log_probs_batch,
                 adv_targ,
                 available_actions_batch) = sample[i]

                preds.append(value_preds_batch)
                rets.append(return_batch)
                a_masks.append(active_masks_batch)
                acts.append(actions_batch)
                s_obs.append(share_obs_batch)




                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy = self.policy[i].evaluate_actions(cent_obs=share_obs_batch,
                                                                                      obs=np.expand_dims(obs_batch, axis=1),
                                                                                      rnn_states_actor=np.expand_dims(rnn_states_batch, axis=1),
                                                                                      rnn_states_critic=rnn_states_critic_batch,
                                                                                      action=np.expand_dims(actions_batch, axis=1),
                                                                                      masks=np.expand_dims(masks_batch,axis=1),
                                                                                      available_actions=np.expand_dims(available_actions_batch,axis=1),
                                                                                      )  # cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                # available_actions=None, active_masks=None
                # actor update
                mix_vals.append(values)

                imp_weights = th.exp(action_log_probs.squeeze(1) - torch.from_numpy(old_action_log_probs_batch).to(self.device))

                surr1 = imp_weights * torch.from_numpy(adv_targ).to(self.device)
                surr2 = th.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * torch.from_numpy(adv_targ).to(self.device)

                if self._use_policy_active_masks:
                    policy_action_loss = (-th.sum(th.min(surr1, surr2),
                                                  dim=-1,
                                                  keepdim=True) * torch.from_numpy(
                        active_masks_batch).to(self.device)).sum() / torch.from_numpy(active_masks_batch).to(self.device).sum()
                else:
                    policy_action_loss = -th.sum(th.min(surr1, surr2), dim=-1, keepdim=True).mean()

                policy_loss = th.clamp( policy_action_loss, min = -100, max = 100)

                self.policy[i].actor_optimizer[0].zero_grad()

                if update_actor:
                    (policy_loss - dist_entropy * self.entropy_coef).backward()

                if self._use_max_grad_norm:
                    actor_grad_norm = th.nn.utils.clip_grad_norm_(self.policy[i].actor[0].parameters(), 10)

                else:
                    actor_grad_norm = get_gard_norm(self.policy[i].actor[0].parameters())

                self.policy[i].actor_optimizer[0].step()

                policy_losses.append(policy_loss.cpu().detach())
                dist_entropies.append(dist_entropy.cpu().detach())
                actor_grad_norms.append(actor_grad_norm.cpu().detach())
                imp_weight.append(imp_weights.cpu().detach())




            if self.mixer is not None:




                share_obs_batch = torch.from_numpy(np.stack(s_obs)).to(self.device)
                actions_batch = torch.from_numpy(np.stack(acts)).to(self.device)
                value_preds_batch = torch.from_numpy(np.stack(preds)).to(self.device)

                if self.mixer.name == "qatten":
                    preds,pv_mags, pv_ents = self.mixer(torch.stack(mix_vals).permute(dims=(1,0,2)),
                                                          share_obs_batch.permute(dims=(1,0,2)),
                                                          actions_batch.permute(dims=(1,0,2)))

                    mix_vals,  v_mags, v_ents  = self.mixer(value_preds_batch.permute(dims=(1,0,2)),
                                                         share_obs_batch.permute(dims=(1,0,2)),
                                                         actions_batch.permute(dims=(1,0,2)))

                elif self.mixer.name == "qmix":
                    preds = self.mixer(value_preds_batch, share_obs_batch)
                    mix_vals = self.mixer(torch.stack(mix_vals), share_obs_batch)

                else:
                    preds = self.mixer(value_preds_batch, share_obs_batch)
                    mix_vals = self.mixer(torch.stack(mix_vals), share_obs_batch)

            # critic update
            # iterate agents
            cum_val = torch.Tensor([0.0]).to(self.policy[0].device)
            mix_vals = torch.stack(mix_vals) if type(mix_vals) is list else mix_vals
            rets = np.stack(rets) if type(rets) is list else rets
            preds = np.stack(preds) if type(preds) is list else preds
            a_masks = np.stack(a_masks) if type(a_masks) is list else a_masks

            if type(preds) is list:
                preds = torch.from_numpy(np.stack(preds)).to(self.device)

            for i in range(self.n_agents):
                value_loss = self.cal_value_loss(mix_vals[i],

                                                 torch.from_numpy(preds[i]).to(self.device),

                                                 torch.from_numpy(rets[i]).to(self.device),

                                                 a_masks[i],

                                                 i) #preds = old, mix = new
                value_losses.append(value_loss)


                if self.mixer is None:
                    self.policy[i].critic_optimizer.zero_grad()

                else:
                    cum_val += (value_loss * self.value_loss_coef)/self.n_agents


            if self.mixer is None:
                (torch.stack(value_losses).mean() * self.value_loss_coef).backward()

                for i in range(self.n_agents):


                    if self._use_max_grad_norm:
                        critic_grad_norm = th.nn.utils.clip_grad_norm_(self.policy[i].critic.parameters(),
                                                                       self.max_grad_norm)

                    else:
                        critic_grad_norm = get_gard_norm(self.policy[i].critic.parameters())

                    self.policy[i].critic_optimizer.step()

                    critic_grad_norms.append(critic_grad_norm)
                    value_losses[i] = value_losses[i].cpu().detach().numpy()

            else:
                for i in range(self.n_agents):
                    self.policy[i].critic_optimizer.zero_grad()
                (cum_val).backward()
                for i in range(self.n_agents):
                    if self._use_max_grad_norm:
                        critic_grad_norm = th.nn.utils.clip_grad_norm_(self.policy[i].critic.parameters(),
                                                                       self.max_grad_norm)

                    else:
                        critic_grad_norm = get_gard_norm(self.policy[i].critic.parameters())

                    self.policy[i].critic_optimizer.step()
                    critic_grad_norms.append(critic_grad_norm.cpu().detach())


            return value_losses, critic_grad_norms, policy_losses, dist_entropies, actor_grad_norms, imp_weight

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch, index = None):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """

        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)
        if self._use_valuenorm or self._use_popart:

            if self.shared:
                self.value_normalizer.update(return_batch)
                error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
                error_original = self.value_normalizer.normalize(return_batch) - values

            else:

                self.value_normalizer[index].update(return_batch)
                error_clipped = self.value_normalizer[index].normalize(return_batch) - value_pred_clipped
                error_original = self.value_normalizer[index].normalize(return_batch) - values

        else:
            error_clipped = torch.from_numpy(return_batch) - value_pred_clipped
            error_original = torch.from_numpy(return_batch) - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)

        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = th.max(value_loss_original, value_loss_clipped)

        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()

        else:
            value_loss = value_loss.mean()

        return value_loss

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.logger.console_logger.info("Updated target network")

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


    def _compute_policy_entropy(self, obs, selected_actions, rnn_states, masks):
        r"""Compute entropy value of probability distribution.

        Notes: P is the maximum episode length (self.max_episode_length)

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N, P, O*)`.

        Returns:
            torch.Tensor: Calculated entropy values given observation
                with shape :math:`(N, P)`.

        """

        entropies = []

        if type(self.policy) is not list:

            policy_entropy = self.policy.get_entropy(obs,
                                                        selected_actions,
                                                        rnn_states,  masks)

            if self._use_softplus_entropy:
                policy_entropy = torch.nn.functional.softplus(policy_entropy)

        else:

            entropies.append(self.policy[0].get_entropy(obs ,
                                                            selected_actions,
                                                            rnn_states, masks)[0].entropy())

            if self._use_softplus_entropy:
                entropies = torch.nn.functional.softplus(entropies)

            policy_entropy = entropies




        return policy_entropy

def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def mse_loss(e):
    return e**2/2