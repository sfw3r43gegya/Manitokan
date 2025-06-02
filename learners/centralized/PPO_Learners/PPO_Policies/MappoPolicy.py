import torch
from modules.critics.ppo import PPOCritic as R_Critic
from modules.agents.actor_agent import R_Actor
from redistributions.conspecfunction.a2c_ppo_acktr.modelRL import CNNBase, MLPBase
import numpy as np

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.act_space = act_space
        self.obs_space = obs_space
        self.encoder = None
        self.use_cnn = args.use_cnn
        self.central_actor = args.one_actor
        self.last_action_obs = args.last_action_obs

        self.n_agents = args.n_agents if args.share_buffer else 1

        if args.use_encoder and self.use_cnn: # recurrent, recurrent_input_size, hidden_size
            if type(args.obs_shape) is int:

                shape = [obs_space]

                self.encoder = CNNBase(
               6,
                args.use_recurrent_policy,
                hidden_size=args.critic_hidden_dim,
            )
            else:


                shape = obs_space


                self.encoder = CNNBase(
                6,
                args.use_recurrent_policy,
                hidden_size=args.critic_hidden_dim,
                )

            self.obs_space = shape

        elif args.use_encoder:
            if args.share_buffer:
                shape = cent_obs_space
            else:
                shape = obs_space
            self.encoder = MLPBase(
                shape,
                args.use_recurrent_policy,
                hidden_size=args.critic_hidden_dim,
                )

        else:
            self.encoder = None
            self.obs_space = obs_space


        self.share_obs_space = cent_obs_space

        self.critic = R_Critic(args=args, cent_obs_space=self.share_obs_space, device=self.device)

        if args.obs_last_action:
            self.obs_space += self.act_space

        self.actor =  [R_Actor(args,
                               self.obs_space,
                               self.act_space,
                               self.device) for i in range(self.n_agents)] if not args.one_actor else R_Actor(args,
                                                                                                              self.share_obs_space,
                                                                                                              self.act_space,
                                                                                                              self.device)



        self.actor_optimizer = [torch.optim.Adam(self.actor[i].parameters(),
                                                lr=self.lr,
                                                 eps=self.opti_eps,
                                                weight_decay=self.weight_decay)for i in range(self.n_agents)] if not args.one_actor else torch.optim.Adam(self.actor.parameters(),
                                                                                                                                                          lr=self.lr,
                                                                                                                                                          eps=self.opti_eps,
                                                                                                                                                          weight_decay=self.weight_decay)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)



    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        if type(self.actor) is list:
            for i in range(self.n_agents):
                update_linear_schedule(self.actor_optimizer[i], episode, episodes, self.lr)
        else:
            update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)

        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def select_actions(self,
                       cent_obs,
                       obs,
                       rnn_states_actor,
                       rnn_states_critic,
                       masks,
                       available_actions=None,
                       deterministic=False,
                       iter_over=None,
                       last_action=None):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """


        acts = []
        probs = []
        states = []
        if not iter_over:
            iter_over = range(self.n_agents)

        if type(self.actor) is list:
            for i in iter_over:
                avals = available_actions.reshape(self.n_agents, -1, self.act_space)[
                    i] if available_actions is not None else None
                if last_action is not None:

                    actions, action_log_probs, rnn_states_actors = self.actor[i](last_action[:, i],
                                                                                 np.concatenate(rnn_states_actor[:, i]),
                                                                                 np.concatenate(masks[:, i]),
                                                                                 avals,
                                                                                 deterministic)

                else:


                    actions, action_log_probs, rnn_states_actors = self.actor[i](obs[:,i],
                                                                     np.concatenate(rnn_states_actor[:,i]),
                                                                     np.concatenate(masks[:,i]),
                                                                     avals,
                                                                     deterministic)
                acts.append(actions)
                probs.append(action_log_probs)
                states.append(rnn_states_actors)
            rnn_states_critic = np.concatenate(rnn_states_critic).squeeze()
        else:
            avals = available_actions.reshape(self.n_agents, -1, self.act_space) if available_actions is not None else None

            if last_action is not None:


                actions, action_log_probs, rnn_states_actors = self.actor(last_action.reshape(obs.shape[0]*self.n_agents, -1),
                                                                          rnn_states_actor.reshape(obs.shape[0]*self.n_agents, -1),
                                                                          masks.reshape(obs.shape[0]*self.n_agents, -1).swapaxes(0,1),
                                                                         avals.reshape(obs.shape[0]*self.n_agents, -1),
                                                                         deterministic)
            else:
                actions, action_log_probs, rnn_states_actors = self.actor(
                    cent_obs.reshape(obs.shape[0] * self.n_agents, -1),
                    rnn_states_actor.reshape(obs.shape[0] * self.n_agents, -1),
                    masks.reshape(obs.shape[0] * self.n_agents, -1).swapaxes(0, 1),
                    avals.reshape(obs.shape[0] * self.n_agents, -1),
                    deterministic)

            acts.append(actions)
            probs.append(action_log_probs)
            states.append(rnn_states_actors)

            rnn_states_critic =rnn_states_critic.reshape(obs.shape[0]*self.n_agents, -1).squeeze()



        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, np.concatenate(masks))

        return values, torch.stack(acts), torch.stack(probs), torch.stack(states), rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        if self.encoder:
            tmp = torch.from_numpy(cent_obs)
            if self.use_cnn:
                tmp = tmp.reshape(tmp.shape[0], 6, 3, 3)
            latents, _, hiddens = self.encoder(inputs=tmp,
                                               rnn_hxs=torch.from_numpy(rnn_states_critic).squeeze(),
                                               masks=torch.from_numpy(masks))
            cent_obs = latents
            rnn_states_critic = hiddens.reshape(shape=rnn_states_critic.shape)

        values, _ = self.critic(cent_obs, rnn_states_critic.squeeze(), masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None, iter_over=None,new_obs=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """

        diss = []
        probs = []

        if not iter_over:
            iter_over = range(self.n_agents)

        if type(self.actor) is list:
            for i in iter_over:
                avals = available_actions[:,i] if available_actions is not None else None
             #   if len(rnn_states_actor.shape) == 4:
                    #rnn_states_actor = rnn_states_actor.squeeze(1)
                if self.last_action_obs:
                    action_log_probs, dist_entropies = self.actor[i].select_actions(obs=new_obs[:, i],
                                                                                    rnn_states=rnn_states_actor[:, i],
                                                                                    action=action[:, i],
                                                                                    masks=masks[:, i],
                                                                                    available_actions=avals,
                                                                                    active_masks=active_masks)

                else:
                    action_log_probs, dist_entropies = self.actor[i].select_actions(obs=obs[:,i],
                                                                         rnn_states=rnn_states_actor[:,i],
                                                                         action=action[:,i],
                                                                         masks=masks[:,i],
                                                                         available_actions=avals,
                                                                        active_masks= active_masks)

                diss.append(dist_entropies)
                probs.append(action_log_probs)
        else:

            if self.last_action_obs:

#
                avals = available_actions if available_actions is not None else None
                #   if len(rnn_states_actor.shape) == 4:
                # rnn_states_actor = rnn_states_actor.squeeze(1)
                action_log_probs, dist_entropies = self.actor.select_actions(obs=new_obs.reshape(obs.shape[0]*self.n_agents, -1),
                                                                                rnn_states=np.expand_dims(rnn_states_actor.reshape(rnn_states_actor.shape[0]*self.n_agents, -1),1),
                                                                                action=action.reshape(obs.shape[0]*self.n_agents, -1),
                                                                                masks=masks.reshape(obs.shape[0]*self.n_agents, -1),
                                                                                available_actions=avals.reshape(obs.shape[0]*self.n_agents, -1),
                                                                                active_masks=active_masks)
            else:

                avals = available_actions if available_actions is not None else None
                #   if len(rnn_states_actor.shape) == 4:
                # rnn_states_actor = rnn_states_actor.squeeze(1)
                action_log_probs, dist_entropies = self.actor.select_actions(obs=cent_obs.reshape(obs.shape[0]*self.n_agents, -1),
                                                                                rnn_states=np.expand_dims(rnn_states_actor.reshape(rnn_states_actor.shape[0]*self.n_agents, -1),1),
                                                                                action=action.reshape(obs.shape[0]*self.n_agents, -1),
                                                                                masks=masks.reshape(obs.shape[0]*self.n_agents, -1),
                                                                                available_actions=avals.reshape(obs.shape[0]*self.n_agents, -1),
                                                                                active_masks=active_masks)


            diss.append(dist_entropies)
            probs.append(action_log_probs)


        values, _ = self.critic(cent_obs.reshape(-1, cent_obs.shape[-1]), rnn_states_critic, np.concatenate(masks))

        return values, torch.stack(probs).swapaxes(0,1), torch.stack(diss)


    def get_entropy(self, obs, aval, rnn_states, masks):

        dists = []

        if type(self.actor) is list:
            for i in range(self.n_agents):
                #   if len(rnn_states_actor.shape) == 4:
                # rnn_states_actor = rnn_states_actor.squeeze(1)
                entropy = self.actor[i].get_entropy(obs=obs[:,:,i].reshape(-1, obs.shape[3]),
                                                    aval=aval[:,:,i].reshape(-1, aval.shape[3]),
                                                    rnn_states=np.expand_dims(rnn_states[:,:,i].reshape(-1, rnn_states.shape[4]),axis=1)[::10,:],
                                                    masks=masks[:,:,i].reshape(-1, masks.shape[3]))

                dists.append(entropy.entropy())

            dists = torch.stack(dists)

        else:


            #   if len(rnn_states_actor.shape) == 4:
            # rnn_states_actor = rnn_states_actor.squeeze(1)
            dist = self.actor.get_entropy(obs.reshape(-1, obs.shape[3]),
                                          aval.reshape(-1, aval.shape[3]),
                                          np.expand_dims(rnn_states.reshape(-1, rnn_states.shape[4]),axis=1)[::10,:],
                                          masks.reshape(-1, masks.shape[3]))
            dists.append(dist.entropy())
            dists = dists[0]

        return dists

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False,iter_over=None):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        acts = []
        states = []

        if not iter_over:
            iter_over = range(self.n_agents)

        if type(self.actor) is list:
            for i in iter_over:
                avals = available_actions[:,i] if available_actions is not None else None
                actions, _, rnn_states_actor = self.actor[i](obs[:,i], rnn_states_actor[:,i], masks[:,i], avals, deterministic)

                acts.append(actions)
                states.append(rnn_states_actor)
        else:
            avals = available_actions if available_actions is not None else None
            actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, avals,
                                                         deterministic)

            acts.append(actions)
            states.append(rnn_states_actor)

        return torch.stack(acts), torch.stack(states)