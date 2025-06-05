import os
from copy import deepcopy

import numpy as np
import torch
from torch.distributions import Categorical
from modules.agents.dqn_agent import RNNActorModel
from modules.mixers import REGISTRY as mix_REGISTRY
from .base_agent import BaseAgent

from .utils import (LinearDecayScheduler, MultiStepScheduler,
                    hard_target_update,
                               soft_target_update)

# https://github.com/jianzhnie/deep-marl-toolkit/tree/main for qmix, vdn, qtran
# https://github.com/WentseChen/Soft-QMIX/blob/master/src/learners/nq_learner.py entropy regularization

class QMixAgent(BaseAgent): # can use, QMIX and VDN mixers
    """QMIX algorithm
    Args:
        actor_model (nn.Model): agents' local q network for decision making.
        mixer_model (nn.Model): A mixing network which takes local q values as input
            to construct a global Q network.
        double_q (bool): Double-DQN.
        gamma (float): discounted factor for reward computation.
        lr (float): learning rate.
        clip_grad_norm (None, or float): clipped value of gradients' global norm.
    """

    def __init__(
        self,
  args ) -> None:

       # check_model_method(actor_model, 'init_hidden', self.__class__.__name__)
       # check_model_method(actor_model, 'forward', self.__class__.__name__)


        self.num_envs = int(args.batch_size_run)
        self.num_agents = int(args.n_agents)
        self.double_q = args.double_q
        self.gamma = float(args.gamma)
        self.optimizer_type = args.optimizer_type
        self.learning_rate = float(args.learning_rate)
        self.min_learning_rate = args.min_learning_rate
        self.clip_grad_norm = args.clip_grad_norm
        self.batch_size = int(args.batch_size_run)
        self.n_actions = int(args.n_actions)

        self.last_action_obs = args.obs_last_action
        if self.last_action_obs:
            args.obs_shape += args.n_actions
        self.obs_shape = args.obs_shape
        self.global_steps = 0
        self.exploration = float(args.egreedy_exploration)
        self.min_exploration = float(args.min_exploration)
        self.target_update_count = 0
        self.target_update_tau = args.target_update_tau
        self.target_update_interval = int(args.target_update_interval)
        self.learner_update_freq = args.learner_update_freq
        self.hidden_size = int(args.rnn_hidden_dim)
        self.recurrent_N = args.recurrent_N
        self.sample_size = args.batch_size
        self.load_models = args.load_model
        self.max_steps = int( args.env_args["episode_limit"])
        self.ten = args.ten



        self.device = args.device
        self.actor_model = RNNActorModel(args) # create an actor
        self.target_actor_model = deepcopy(self.actor_model)
        self.actor_model.to(args.device)
        self.target_actor_model.to(args.device)
        self.eps_limit = float(args.eps_limit)


        self.entropy_coef=args.policy_ent_coeff
        self.entropy_method = args.entropy_method

        self.params = list(self.actor_model.parameters())

        self.mixer_model = None
        if args.mixer is not None and args.mixer != '' and args.mixer != "''":
            if args.mixer == "maven":
                self.mixer_model = mix_REGISTRY[args.mixer](args=args, )
            else:
                self.mixer_model = mix_REGISTRY[args.mixer](args = args, cent_obs_dim =args.obs_shape*args.n_agents ,num_agents=args.n_agents, device=args.device)
            self.target_mixer_model = deepcopy(self.mixer_model)
            self.mixer_model.to(args.device)
            self.target_mixer_model.to(args.device)
            self.params += list(self.mixer_model.parameters())

        if self.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(params=self.params,
                                              lr=self.learning_rate)
        else:
            self.optimizer = torch.optim.RMSprop(
                params=self.params,
                lr=self.learning_rate,
                alpha=args.optim_alpha,
                eps=args.optim_eps,
            )

        if not self.ten:
            self.ep_scheduler = LinearDecayScheduler(args.egreedy_exploration,
                                                 args.t_max* self.eps_limit)
            lr_milstons = [args.t_max * 0.5, args.t_max * 0.8]
            self.lr_scheduler = MultiStepScheduler(
                start_value=args.learning_rate,
                max_steps=args.t_max,
                milestones=lr_milstons,
                decay_factor=0.1,
            )

        else:
            self.ep_scheduler = LinearDecayScheduler(args.egreedy_exploration,
                                                     int(3000000 * self.eps_limit))
            lr_milstons = [3000000 * 0.5, 3000000 * 0.8]
            self.lr_scheduler = MultiStepScheduler(
                start_value=args.learning_rate,
                max_steps=3000000,
                milestones=lr_milstons,
                decay_factor=0.1,
            )





        # 执行过程中，要为每个agent都维护一个 hidden_state
        # 学习过程中，要为每个agent都维护一个 hidden_state、target_hidden_state
        self.hidden_state = None
        self.target_hidden_state = None

    def init_hidden_states(self, batch_size: int = 1) -> None:
        """Initialize hidden states for each agent.

        Args:
            batch_size (int): batch size
        """
        self.hidden_state = torch.zeros((1, self.hidden_size, batch_size, self.num_agents, self.recurrent_N,),
                                        device=self.device)


        self.target_hidden_state =  torch.zeros((1, self.hidden_size,batch_size, self.num_agents, self.recurrent_N,),
                                        device=self.device)


    def sample(self, obs: torch.Tensor,
               available_actions: torch.Tensor) -> np.ndarray:
        """sample actions via epsilon-greedy
        Args:
            obs (np.ndarray):               (num_agents, obs_shape)
            available_actions (np.ndarray): (num_agents, n_actions)
        Returns:
            actions (np.ndarray): sampled actions of agents
        """
        epsilon = np.random.random()
        if epsilon < self.exploration:
            available_actions = torch.tensor(available_actions)

            actions_dist = Categorical(available_actions)
            actions = actions_dist.sample().numpy()

        else:
            actions = self.predict(obs.reshape(self.batch_size*self.num_agents,self.obs_shape ), available_actions)

        # update exploration
        self.exploration = max(self.ep_scheduler.step(), self.min_exploration)

        return actions

    def predict(self, obs: torch.Tensor,
                available_actions: torch.Tensor) -> np.ndarray:
        """take greedy actions
        Args:
            obs (np.ndarray):               (num_agents, obs_shape)
            available_actions (np.ndarray): (num_agents, n_actions)
        Returns:
            actions (np.ndarray):           (num_agents, )
        """
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).squeeze()
        available_actions = torch.tensor(available_actions,
                                         dtype=torch.long,
                                         device=self.device)#.squeeze()

        agents_q, self.hidden_state = self.actor_model(obs, self.hidden_state)

        # mask unavailable actions
        agents_q = agents_q.reshape(available_actions.shape)
        agents_q[available_actions == 0] = 1e-10
        actions = agents_q.argmax(dim=2).detach().cpu().numpy()
        return [actions]

    def update_target(self, target_update_tau: float = 0.05) -> None:
        """Update target network with the current network parameters.

        Args:
            target_update_tau (float): target update tau
        """
        if target_update_tau:
            soft_target_update(self.actor_model, self.target_actor_model,
                               target_update_tau)

            if self.mixer_model:
                soft_target_update(self.mixer_model, self.target_mixer_model,
                               target_update_tau)
        else:
            hard_target_update(self.actor_model, self.target_actor_model)
            if self.mixer_model:
                hard_target_update(self.mixer_model, self.target_mixer_model)

    def learn(self, episode_data):
        """Update the model from a batch of experiences.

        Args: episode_data (dict) with the following:

            - obs (np.ndarray):                     (batch_size, T, num_agents, obs_shape)
            - state (np.ndarray):                   (batch_size, T, state_shape)
            - actions (np.ndarray):                 (batch_size, T, num_agents)
            - rewards (np.ndarray):                 (batch_size, T, 1)
            - dones (np.ndarray):                   (batch_size, T, 1)
            - available_actions (np.ndarray):       (batch_size, T, num_agents, n_actions)
            - filled (np.ndarray):                  (batch_size, T, 1)

        Returns:
            - mean_loss (float): train loss
            - mean_td_error (float): train TD error
        """
        # get the data from episode_data buffer
        obs_episode = episode_data['obs'].squeeze()

        actions_episode = episode_data['actions']
        available_actions_episode = episode_data['available_actions'].squeeze()
        rewards_episode = episode_data['rewards'].squeeze(3)
        dones_episode = episode_data['dones']
        filled_episode = episode_data['filled']

        if self.last_action_obs:
            tmp = []
            for i in range(obs_episode.shape[1]):

                if i == 0:
                    o = obs_episode[:, i]
                    a =  torch.zeros_like(available_actions_episode[:,i])
                    tmp.append(torch.cat([o, a],dim=len(o.shape)-1))
                else:
                    o = obs_episode[:, i]
                    a =  torch.eye(self.n_actions)
                    a = a[actions_episode[:, i-1].long()]

                    if len(o.shape) < len(a.shape):
                        a = a.squeeze()

                    tmp.append(torch.cat([o, a],dim=len(o.shape)-1) )

            obs_episode = torch.stack(tmp).swapaxes(0,1)
        # update target model
        if self.global_steps % self.target_update_interval == 0:
            self.update_target(self.target_update_tau)
            self.target_update_count += 1

        self.global_steps += 1

        # set the actions to torch.Long
        actions_episode = actions_episode.to(torch.long).to(self.device).clone().detach()

        # get the batch_size and episode_length


        # get the relevant quantitles
        actions_episode = actions_episode[:, :-1, :].unsqueeze(-1)
        actions_episode = torch.nn.functional.one_hot(actions_episode).squeeze()
        rewards_episode = rewards_episode[:, :-1, :]
        dones_episode = dones_episode[:, :-1, :].float()
        filled_episode = filled_episode[:, :-1, :].float()

        mask = (1 - dones_episode) * (1 - filled_episode)

        # Calculate estimated Q-Values
        local_qs = []
        target_local_qs = []
        self.init_hidden_states(self.sample_size*self.batch_size)
        hid_states = []
        t_hid_states = []
        for t in range(self.max_steps):
            if self.sample_size > 1 and self.num_agents > 1:
                obs = obs_episode[:, t, :, :]
            elif self.sample_size > 1:
                obs = obs_episode[:, t, :]
            else:
                obs = obs_episode[ t, :, :]
            # obs: (batch_size * num_agents, obs_shape)
            obs = obs.reshape(-1, obs.shape[-1])
            # Calculate estimated Q-Values
            local_q, self.hidden_state = self.actor_model(
                obs, self.hidden_state)
            # local_q: (batch_size * num_agents, n_actions) -->  (batch_size, num_agents, n_actions)
            local_q = local_q.reshape(self.sample_size*self.batch_size, self.num_agents, -1)
            local_qs.append(local_q)
            hid_states.append(self.hidden_state)

            # Calculate the Q-Values necessary for the target
            target_local_q, self.target_hidden_state = self.target_actor_model(
                obs, self.target_hidden_state)
            t_hid_states.append(self.target_hidden_state)
            # target_local_q: (batch_size * num_agents, n_actions) -->  (batch_size, num_agents, n_actions)
            target_local_q = target_local_q.view(self.sample_size*self.batch_size, self.num_agents,
                                                 -1)
            target_local_qs.append(target_local_q)

        # Concat over time
        local_qs = torch.stack(local_qs, dim=1)
        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_local_qs = torch.stack(target_local_qs[1:], dim=1)

        # Pick the Q-Values for the actions taken by each agent
        # Remove the last dim
        if self.sample_size > 1 and self.batch_size > 1 or self.num_agents > 1 and self.batch_size != 1:
            indices = torch.argmax(actions_episode,dim =4, keepdim=True)
            indices = indices.reshape(self.sample_size*self.batch_size,indices.shape[1], self.num_agents, 1)

            chosen_action_local_qs = torch.gather(local_qs[:, :-1, :],
                                                  dim=3,
                                                  index=indices).squeeze(0)

        elif self.sample_size > 1:
            indices = torch.argmax(actions_episode, dim=3, keepdim=True)#.unsqueeze(2)
            chosen_action_local_qs = torch.gather(local_qs[:, :-1, :, :],
                                                  dim=3,
                                                  index=indices)

        else:
            indices = torch.argmax(actions_episode, dim =2, keepdim=True).unsqueeze(2).swapaxes(1, 2).swapaxes(0,1)
            chosen_action_local_qs = torch.gather(local_qs[:, :-1, :, :],
                                                  dim=3,
                                                  index=indices).squeeze(3)



        # mask unavailable actions

        if self.num_agents > 1:
            target_local_qs[available_actions_episode[:, 1:,:,:].reshape(target_local_qs.shape) == 0] = -1e10

        elif self.sample_size > 1 and self.batch_size > 1:
            target_local_qs[available_actions_episode[:, 1:, :].reshape(self.sample_size*self.batch_size, available_actions_episode.shape[1]-1, -1) .unsqueeze(2) == 0] = 1e10
        elif self.sample_size > 1:
            target_local_qs[available_actions_episode[:, 1:, :].unsqueeze(2) == 0] = -1e10
        else:
            target_local_qs[available_actions_episode[1:, :, :].unsqueeze(2).swapaxes(0,1) == 0] = -1e10

        # Max over target Q-Values
        if self.double_q:
            # Get actions that maximise live Q (for double q-learning)
            local_qs_detach = local_qs.clone().detach()
            local_qs_detach[available_actions_episode == 0] = -1e10
            cur_max_actions = local_qs_detach[:, 1:].max(dim=3,
                                                         keepdim=True)[1]
            target_local_max_qs = torch.gather(
                target_local_qs, dim=3, index=cur_max_actions).squeeze(3)
        else:
            # idx0: value, idx1: index
            target_local_max_qs = target_local_qs.max(dim=3)[0]

        # Mixing network
        # mix_net, input: ([Q1, Q2, ...], state), output: Q_total
        if self.mixer_model is not None:


            if self.mixer_model.name == "qtrans":
                chosen_action_global_qs,  chosen_v_outputs = self.mixer_model(
                episode_data, hidden_states = torch.cat(hid_states),
                    batch_size=self.sample_size*self.batch_size,
                    max_seq_length =obs_episode.shape[1])



                target_global_max_qs, target_v_outputs = self.target_mixer_model(
                episode_data, hidden_states = torch.cat(t_hid_states), batch_size=self.sample_size*self.batch_size, max_seq_length =obs_episode.shape[1])


            else:
                chosen_action_global_qs = self.mixer_model(
                chosen_action_local_qs, states = obs_episode[:, :-1, :,:].reshape( self.num_agents,
                                                                                 obs_episode[:, :-1, :,:].shape[1],
                                                                                self.batch_size*self.sample_size, -1))
                target_local_max_qs = target_local_max_qs.unsqueeze(-1)
                target_global_max_qs = self.target_mixer_model(
                target_local_max_qs, states = obs_episode[:, 1:, :,:].reshape( self.num_agents,
                                                                                 obs_episode[:, :-1, :,:].shape[1],
                                                                                self.batch_size*self.sample_size, -1))

        if self.mixer_model is None:
            target_max_qvals = target_local_max_qs
            chosen_action_qvals = chosen_action_local_qs
        else:

            if self.mixer_model.name != "qtrans":
                target_max_qvals = target_global_max_qs.reshape(rewards_episode.shape)
                chosen_action_qvals = chosen_action_global_qs.reshape(rewards_episode.shape)

            else:
                target_max_qvals = target_global_max_qs.reshape(rewards_episode.shape[0],
                                                                rewards_episode.shape[1]+1,
                                                                rewards_episode.shape[2],
                                                                rewards_episode.shape[3])
                target_max_qvals = target_max_qvals[:,1:,:,:]

                chosen_action_qvals = chosen_action_global_qs.reshape(rewards_episode.shape[0],
                                                                      rewards_episode.shape[1]+1,
                                                                      rewards_episode.shape[2],
                                                                      rewards_episode.shape[3])
                chosen_action_qvals = chosen_action_qvals[:,:-1,:,:]

        # Calculate 1-step Q-Learning targets
        if self.entropy_method == "max" :

            q_vals_ent = local_qs.clone().detach()[:, 1:, :]

            if self.mixer_model.name in ["vdn", "qmix"]:
                q_vals_ent = self.mixer_model(
                    episode_data, states=obs_episode[:, :-1, :, :].reshape(self.num_agents,
                                                                                     obs_episode[:, :-1, :, :].shape[1],
                                                                                     self.batch_size * self.sample_size,
                                                                                     -1))
            else:

                _, q_vals_ent = self.mixer_model(
                    episode_data, hidden_states=torch.cat(hid_states),
                    batch_size=self.sample_size * self.batch_size,
                    max_seq_length=obs_episode.shape[1], ent =q_vals_ent)


            idx_act = q_vals_ent.shape[len(q_vals_ent.shape)-1]
            q_vals_ent = q_vals_ent.view(available_actions_episode[:, 1:, :] .shape)
            q_vals_ent = q_vals_ent/ self.entropy_coef
            q_vals_ent[available_actions_episode[:, 1:, :]  == 0] = -1e10

            ent_pdf = torch.softmax(q_vals_ent, dim=idx_act)

            rand_idx = torch.rand(ent_pdf.shape).to(ent_pdf.device)
            actions_cdf = torch.cumsum(ent_pdf, idx_act)
            rand_idx = torch.clamp(rand_idx, 1e-6, 1-1e-6)
            picked_actions = torch.searchsorted(actions_cdf, rand_idx)
            #ent_pdf[ent_pdf==0] = 0.0000000000001
            target_logp = torch.log(ent_pdf)
            target_logp = torch.gather(target_logp, idx_act, picked_actions).squeeze(idx_act)
            target_entropy = -target_logp.sum(idx_act,keepdim=True)*self.entropy_coef

        else:
            target_entropy = 0

        if self.num_agents > 1:
            target = rewards_episode + target_entropy+ self.gamma * (
            1 - dones_episode.unsqueeze(3)) * target_max_qvals

        elif self.sample_size > 1 and self.batch_size > 1:
            target = rewards_episode[:, :, :, 0].reshape(self.sample_size*self.batch_size, rewards_episode.shape[1], -1 ).squeeze()+ target_entropy + self.gamma * (
                    1 - dones_episode.reshape(self.sample_size*self.batch_size, -1).squeeze() ) * target_max_qvals.squeeze()

        elif self.sample_size > 1:
            target = rewards_episode[:, :, :, 0].squeeze() + target_entropy+ self.gamma * (
                        1 - dones_episode.squeeze()) * target_max_qvals.squeeze()

        else:
            target = rewards_episode[:, :, :, 0].squeeze() + target_entropy+ self.gamma * (
                        1 - dones_episode.squeeze()) * target_max_qvals.swapaxes(0, 1).squeeze()
        #  Td-error
        if self.num_agents > 1:
            td_error = target.detach() - chosen_action_qvals
            mask = mask.unsqueeze(3)

        elif self.sample_size > 1 :
            td_error = target.detach() - chosen_action_qvals.squeeze()
            td_error = td_error.unsqueeze(2)
            if self.batch_size > 1:
                mask = mask.reshape(self.sample_size*self.batch_size, -1).unsqueeze(2)

        else:
            td_error = target.detach() - chosen_action_qvals.squeeze().swapaxes(0,1)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        mean_td_error = masked_td_error.sum() / mask.sum()
        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error**2).sum() / mask.sum()

        # Optimise
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip_grad_norm)
        self.optimizer.step()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

        results = {
            'loss': loss.item(),
            'mean_td_error': mean_td_error.item(),
            'average_entropy': target_entropy.mean() if type(target_entropy) is not int else target_entropy,
            'target_update': target.detach().mean().item(),
            'chosen_action_qvals': chosen_action_qvals.squeeze().mean().item()
        }
        return results

    def save_model(
        self,
        save_dir: str = None,
        actor_model_name: str = 'actor_model.th',
        mixer_model_name: str = 'mixer_model.th',
        opt_name: str = 'optimizer.th',
    ):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        actor_model_path = os.path.join(save_dir, actor_model_name)
        mixer_model_path = os.path.join(save_dir, mixer_model_name)
        optimizer_path = os.path.join(save_dir, opt_name)
        torch.save(self.actor_model.state_dict(), actor_model_path)
        torch.save(self.mixer_model.state_dict(), mixer_model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)
        print('save model successfully!')

    def load_model(
        self,
        save_dir: str = None,
        actor_model_name: str = 'actor_model.th',
        mixer_model_name: str = 'mixer_model.th',
        opt_name: str = 'optimizer.th',
    ):
        actor_model_path = os.path.join(save_dir, actor_model_name)
        mixer_model_path = os.path.join(save_dir, mixer_model_name)
        optimizer_path = os.path.join(save_dir, opt_name)
        self.actor_model.load_state_dict(torch.load(actor_model_path))
        self.mixer_model.load_state_dict(torch.load(mixer_model_path))
        self.optimizer.load_state_dict(torch.load(optimizer_path))
        print('restore model successfully!')
