from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import cv2
import torch
import os
from envs.tools.batch_env import  BatchEnv
from copy import deepcopy

from modules.bandits.uniform import Uniform
from modules.bandits.reinforce_hierarchical import EZ_agent as enza
from modules.bandits.returns_bandit import ReturnsBandit as RBandit


def _t2n(x):
    return x.detach().cpu().numpy()




class EpisodeRunner:

    def __init__(self, args,   logger, buffer = None, agent_ids=None, iter_over = None):
        self.args = args
        self.logger = logger
        self.batch_size = int(self.args.batch_size_run)
        self.sample_size = int(self.args.batch_size)


        self.env = BatchEnv(self.batch_size,
                            args.puffer,
                            env_REGISTRY[self.args.env],
                            **self.args.env_args)





        self.episode_limit = int( args.env_args["episode_limit"])
        self.t = 0
        self.total_steps = 0,
        self.episode_step = 0
        self.episode = 0
        self.t_env = 0
        self.count = 0
        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        self.use_centralized_V = args.use_centralized_V
        self.ppo  = True if args.name in ["mappo","ippo"] else False
        self.alg_name = args.name
        self.buffer = buffer
        self.agent_ids = agent_ids
        self.n_rollout_threads = 1
        self.n_agents =  int(args.env_args["n_agents"])
        self.shared = args.share_buffer
        self.central_critic = args.central_critic
        self.n_keys =  args.env_args["n_keys"] if  args.env_args["name"] == "iw2si" else 0
        self.only_sparse = args.env_args['only_sparse']
        self.only_immediate = args.env_args['only_immediate']
        self.reward_coop = args.env_args['reward_coop']
        self.credit_easy_af = args.env_args['credit_easy_af']
        self.iter_over = iter_over
        self.punish_step = args.punish_step


        # Log the first run
        self.statistics_dict = {"reward": [],
                                "success":0,
                                "holds_key": [],
                                "key_dropped": [],
                                "reward_at_step" : [],
                                "agent_distance": [],
                                "key_distance": [],
                                "total_steps": 0,
                                "success_at_step": []
                                }

        self.log_train_stats_t = -1000000

        for _ in range( self.n_agents):
            self.statistics_dict["reward"].append(0)
            self.statistics_dict["holds_key"].append(0)
            self.statistics_dict["reward_at_step"].append(0)
            self.statistics_dict["success_at_step"].append(0)



            for _ in range(self.n_keys):
                self.statistics_dict["key_distance"].append(0)

    def setup(self, scheme, groups, preprocess, mac, buffer):


        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device) if scheme is not None else None

        if self.args.name == "maven":
            if self.args.noise_bandit:
                if self.args.bandit_policy:
                    self.noise_distrib = enza(self.args, logger=self.logger)
                else:
                    self.noise_distrib = RBandit(self.args, logger=self.logger)
            else:
                self.noise_distrib = Uniform(self.args)

            self.noise_returns = {}
            self.noise_test_won = {}
            self.noise_train_won = {}

        self.buffer = buffer
        self.mac = mac

    def get_env_info(self):


        if not self.args.puffer:
            return self.env.get_env_info()

        else:
            info = {}
            info["n_agents"] = int(self.n_agents)
            info["n_actions"] = int(self.args.n_actions)
            info["state_shape"] = int(self.args.state_shape) if int(self.n_agents) > 1 else 78
            info["obs_shape"] = 27 if not self.args.env_args["completion_signal"] else 30

            if self.args.env_args["key_signal"]:
                info["obs_shape"] = info["obs_shape"] + 3
            info["episode_limit"] = self.episode_limit
            return info

   # def save_replay(self):
    #    self.env.save_replay()

    #def close_env(self):
    #    self.env.close()

    def reset(self, test_mode = None):

        if not self.ppo and  self.args.name in [ "coma", "maven","pg"]:
            self.batch = self.new_batch()

        self.env.reset()
        self.t = 0
        self.statistics_dict["reward"] = []
        self.statistics_dict["holds_key"] = []
        self.statistics_dict["key_dropped"] = []
        if self.args.env_args["name"] == "iw2si":
            self.statistics_dict["agent_distance"] = []
        self.statistics_dict["key_distance"] = []
        self.statistics_dict["reward"] = []
        self.statistics_dict["success"] = 0
        self.statistics_dict["reward_at_step"] = []
        self.statistics_dict["success_at_step"] = []
        self.statistics_dict["key_pickup"] = []
        self.statistics_dict["drop_before"] = []
        self.statistics_dict["drop_after"] = []
        self.statistics_dict["key_first"] = []
        self.statistics_dict["door_first"] = []


        if self.args.name == "maven":
            self.noise = self.noise_distrib.sample(self.batch['state'][:, 0], test_mode)

            self.batch.update({"noise": self.noise}, ts=0)

        if self.args.env_args["name"] == "iw2si":
            for _ in range(self.n_agents):
                self.statistics_dict["reward"].append([])
                self.statistics_dict["holds_key"].append([0])
                self.statistics_dict["reward_at_step"].append([0])
                self.statistics_dict["success_at_step"].append([0])
                self.statistics_dict["key_dropped"].append([0 for x in range(self.batch_size)])
                self.statistics_dict["key_pickup"].append([0 for x in range(self.batch_size)])

                self.statistics_dict["drop_before"].append([0 for x in range(self.batch_size)])
                self.statistics_dict["drop_after"].append([0 for x in range(self.batch_size)])
                self.statistics_dict["key_first"].append([0 for x in range(self.batch_size)])
                self.statistics_dict["door_first"].append([0 for x in range(self.batch_size)])






    def run(self, test_mode=False):
        if  self.args.env_args["name"]  == "iw2si":
            if self.args.puffer:
                self.reset(test_mode=test_mode)

                obs = np.array([np.stack(self.env.get_obs())]).swapaxes(0, 1).squeeze()
            else:
                obs = np.array([self.env.get_obs()]).squeeze(0).swapaxes(0, 1).squeeze()



        if self.args.name == "saf":
           self.reset()
           obs, state, act_masks = torch.from_numpy(np.array(self.env.get_obs()).squeeze()), torch.from_numpy(np.array(self.env.get_state())), torch.from_numpy(np.array(self.env.get_avail_actions()))
           next_done = torch.zeros((self.batch_size, self.args.n_agents)).to(self.args.device)

           if self.args.latent_kl:
               ## old_observation - shifted tensor (the zero-th obs is assumed to be equal to the first one)
               obs_old = obs.clone()
               obs_old[1:] = obs_old.clone()[:-1]


               bs = obs_old.shape[0]
               n_ags = obs_old.shape[1]

               obs_old = obs_old.reshape((-1,) + self.mac.policy.obs_shape).to(torch.float32)
               obs_old = self.mac.policy.conv(obs_old)
               obs_old = obs_old.reshape(bs, n_ags, self.mac.policy.input_shape)

        elif not self.args.puffer:
           self.reset(test_mode=test_mode)

        terminated = False
        episode_return = 0
        if self.args.name not in ["mappo","ippo", "qmixer",  "saf", "random"]:
            self.mac.init_hidden(batch_size=self.batch_size)

        elif self.args.name == "qmixer":
            self.mac.init_hidden_states(batch_size=self.batch_size)
        step = 0



        if self.use_centralized_V :
            if self.shared:
                share_obs = obs.reshape(self.batch_size, -1)
            else:
                share_obs = obs.reshape(self.batch_size, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.n_agents, axis=1)
        else:
            share_obs = obs


        if self.shared and self.args.name == "mappo" :
            self.buffer.share_obs[0] = share_obs.squeeze().copy() if self.n_agents > 1 else  share_obs.copy()
            self.buffer.obs[0] = obs.squeeze().copy().swapaxes(0,1) if self.n_agents > 1 else  obs.squeeze(1).copy()
            if self.args.env_args["name"]  == "iw2si":
                self.buffer.available_actions[0] = np.array(np.split(_t2n(torch.Tensor(self.env.get_avail_actions())),
                                                                 self.batch_size)).squeeze() if self.n_agents > 1 else np.array(np.split(_t2n(torch.Tensor(self.env.get_avail_actions())),
                                                                 self.batch_size)).squeeze(1)
            else:
                self.buffer.available_actions[0] = np.array(np.split(_t2n(torch.Tensor(self.env.get_avail_actions())),
                                                                     self.batch_size)).squeeze().swapaxes(1,2)

        elif self.args.name == "ippo":
            for i in range(self.n_agents):
                self.buffer[i].share_obs[0] = share_obs.squeeze()[:,i,:].copy()
                self.buffer[i].obs[0] = obs.squeeze()[i,:,:].copy()
                self.buffer[i].available_actions[0] = np.array(np.split(_t2n(torch.Tensor(self.env.get_avail_actions())),  self.batch_size)).squeeze()[:,i,:]

        train_dic = {}

        if self.args.env_args["name"] == "iw2si":
            for i in range(self.n_agents):
                train_dic["training_keys_dropped_for_agent_" + str(i + 1)] = -1


        last_action = None

        while np.any( terminated == False):


            if self.args.env_args["name"] == "iw2si" and self.args.name != "saf":

                pre_transition_data = {
                    "state": np.array([self.env.get_state()]),
                    "avail_actions": np.array([self.env.get_avail_actions()])[:,:,self.iter_over,:],
                    "obs": obs if   len(obs.shape) == 1 or obs.shape[1] != self.n_agents else obs.swapaxes(1,2),
                }

                if len(pre_transition_data["obs"].shape) == 3:
                    pre_transition_data["obs"] = np.expand_dims(pre_transition_data["obs"].swapaxes(0,1),1)
                pre_transition_data["obs"] = pre_transition_data["obs"][:,:, self.iter_over,:] if self.n_agents >1 and self.batch_size > 1 else  pre_transition_data["obs"]

                pre_transition_data["order"] = np.repeat(np.expand_dims(np.array(self.iter_over), 0),self.batch_size, axis=0)

            elif self.args.env_args["name"]  == "starcraft":
                pre_transition_data = {
                    "state": np.array([self.env.get_state()]).swapaxes(0,1),
                    "avail_actions": np.array([self.env.get_avail_actions()]).swapaxes(0,1).swapaxes(2,3)[:,:,self.iter_over,:],
                    "obs": np.expand_dims(obs,1)[:,:,self.iter_over,:] ,
                }


            else:
                pre_transition_data = None

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1

            if self.alg_name == "random":
                aval = np.array(np.split(_t2n(torch.Tensor(self.env.get_avail_actions())), self.batch_size)).squeeze(
                    1)

                aval = torch.from_numpy(aval).swapaxes(0,1)

                keys_carried = self.env.get_carryings() if self.args.env_args["name"] == "iw2si" else None
                had_key = [[False if x is not None else True for x in l1] for l1 in keys_carried]

                actions = self.mac.act(available_actions=aval)
                reward, terminated = self.env.step(actions, [self.iter_over for _ in range(self.batch_size)] )
                last_action = actions if self.episode_step == 1 else None

                for i in self.iter_over:
                    for j in range(self.batch_size):
                        l = 0
                        if keys_carried[j][i] is not None and actions[j][i] == 4:
                            self.statistics_dict["key_dropped"][i][j] += 1
                            l += 1




            elif self.ppo:
                keys_carried = self.env.get_carryings() if  self.args.env_args["name"]  == "iw2si" else None

                if self.shared:
                    if self.args.env_args["name"] == "iw2si":
                        aval = np.array(
                            np.split(_t2n(torch.Tensor(self.env.get_avail_actions())), self.batch_size)).squeeze(
                            1)
                        aval = aval.reshape(
                            (aval.shape[0] * aval.shape[1], aval.shape[2])) if aval is not None else None

                        keys_carried = [deepcopy(el) for el in self.env.get_carryings()]
                        keys_carried = [keys_carried[kl] for kl in range(len(keys_carried))]

                        for i in range(self.n_agents):
                            l = 0
                            for j in range(self.batch_size):

                                if keys_carried[j][i] is not None and last_action is not None and last_action[:][j][
                                    i] == 3 and had_key[j][i] == True:
                                    self.statistics_dict["key_pickup"][i][j] += 1


                                    if sum(np.array(self.statistics_dict["key_first"])[:,
                                           j]) == 0:  # between all agents in a batch
                                        self.statistics_dict["key_first"][i][j] = 1









                    else:
                        aval = None

                    new_obs=None

                    if self.mac.policy.last_action_obs:
                        new_obs = []
                        ah = []
                        obs =  self.buffer.share_obs[step]

                        if step == 0:
                            acts = np.zeros_like(self.buffer.actions[step])

                        else:
                            acts =  self.buffer.actions[step-1]


                        ah.append(torch.from_numpy(obs))
                        ah.append(torch.from_numpy(acts))
                        new_obs = torch.cat(ah, dim=2)



                    value, actions, action_log_prob, rnn_state, rnn_state_critic = self.mac.policy.select_actions(cent_obs = self.buffer.share_obs[step].reshape(self.batch_size*self.n_agents, -1),
                                                                        obs=self.buffer.obs[step],
                                                                        rnn_states_actor=self.buffer.rnn_states[step],
                                                                        rnn_states_critic= self.buffer.rnn_states_critic[step],
                                                                        masks=self.buffer.masks[step],
                                                                        available_actions = aval,
                                                                        deterministic = test_mode,
                                                                        iter_over=self.iter_over,
                                                                        last_action=new_obs) #   collect

                    value = value.reshape(self.batch_size,  self.n_agents,1,-1)
                    actions = actions.reshape(self.batch_size, self.n_agents, 1, -1)
                    action_log_prob =  action_log_prob.reshape(  self.batch_size, self.n_agents, 1, -1)
                    rnn_state = rnn_state.reshape(  self.batch_size,  self.n_agents, 1, -1)
                    rnn_state_critic = rnn_state_critic.reshape(  self.batch_size,  self.n_agents, 1, -1)
                    last_action = actions

                    if  self.args.env_args["name"]  == "iw2si":
                        had_key = [[False if x is not None else True for x in l1] for l1 in keys_carried]


                           # self.statistics_dict["key_dropped"][i] =  self.statistics_dict["key_dropped"][i]/l if l != 0 else self.statistics_dict["key_dropped"][i]

                    value = np.array(_t2n(value))
                    action_log_prob = np.array(_t2n(action_log_prob))
                    rnn_state = np.array(_t2n(rnn_state))
                    rnn_state_critic = np.array(_t2n(rnn_state_critic))

                    if self.args.env_args["name"] == "iw2si":
                        available_actions = np.array(np.split(_t2n(torch.Tensor(self.env.get_avail_actions())),  self.batch_size))


                    actions_ones = np.eye(self.env._num_actions)[actions.cpu()] if self.args.env_args["name"]  != "forced_coordination" else np.eye(self.env._num_actions.n)[actions.cpu()]

                    actions = [actions]
                    batched_order = np.array([self.iter_over for _ in range(self.batch_size)]).reshape(self.batch_size, -1)
              #      batched_order = np.repeat(self.iter_over, self.batch_size).reshape(self.batch_size, -1)

                    if  self.args.env_args["name"]  == "iw2si":
                        reward, terminated = self.env.step(actions[0].squeeze(), batched_order)  # step
                        reward = reward - 0.0001 if self.punish_step else reward
                        obs = np.array([self.env.get_obs()]).squeeze(0).swapaxes(0, 1)



                    masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in terminated])
                    masks = masks.repeat(1,self.n_agents).unsqueeze(2)
                    bad_masks = masks

                    if self.use_centralized_V:
                        if self.shared:
                            share_obs = obs.reshape(self.batch_size, -1)
                        else:
                            share_obs = obs.reshape(self.batch_size, -1)
                        share_obs = np.expand_dims(share_obs, 1).repeat(self.n_agents, axis=1)
                    else:
                        share_obs = obs




                    data = (share_obs, obs.squeeze() if self.n_agents > 1 else obs.squeeze(1) , rnn_state, rnn_state_critic,
                            actions_ones.squeeze() if self.n_agents > 1 else actions_ones.squeeze(1).squeeze(1), action_log_prob.squeeze(2) ,
                            value.squeeze(2), reward, masks.detach().numpy(), bad_masks, None, available_actions)

                    if not test_mode:
                        share_obs = self.insert(data, iter_over=self.iter_over)  # insert




                else:
                    values = []
                    actions = []
                    action_log_probs = []
                    rnn_states = []
                    rnn_state_critics = []

                    for i in self.iter_over:
                        aval = np.array(
                            np.split(_t2n(torch.Tensor(self.env.get_avail_actions())), self.batch_size)).squeeze()[:, i,
                               :]

                        value, action, action_log_prob, rnn_state, rnn_state_critic = self.mac.policy[i].select_actions(
                            cent_obs=self.buffer[i].share_obs[step],
                            obs=np.expand_dims(self.buffer[i].obs[step], axis =1),
                            rnn_states_actor=np.expand_dims(self.buffer[i].rnn_states[step], axis=1),
                            rnn_states_critic=self.buffer[i].rnn_states_critic[step],
                            masks=np.expand_dims(self.buffer[i].masks[step], axis=1),
                            available_actions=aval,
                            deterministic=test_mode,
                            iter_over = False)  # collect


                        l = 0
                        for j in range(self.batch_size):

                            if keys_carried[j][i] is not None and action[:,j][0] == 4:
                                self.statistics_dict["key_dropped"][i][j] += 1
                                l += 1

                    #    self.statistics_dict["key_dropped"][i] =  self.statistics_dict["key_dropped"][i]/l if l != 0 else self.statistics_dict["key_dropped"][i]

                        values.append(value.detach().cpu().numpy())
                        actions.append( np.array(_t2n(action)))
                        action_log_probs.append(action_log_prob.detach().cpu().numpy())
                        rnn_state = np.array(_t2n(torch.Tensor(rnn_state)))
                        rnn_state_critic = np.array(
                            np.split(_t2n(torch.Tensor(rnn_state_critic)),  self.batch_size))
                        rnn_states.append(rnn_state)
                        rnn_state_critics.append(rnn_state_critic)


                    value = np.array(values)
                    action_log_prob = np.stack(action_log_probs)
                    rnn_state = np.stack(rnn_states)
                    rnn_state_critic = np.stack(rnn_state_critics)
                    available_actions = np.array(np.split(_t2n(torch.Tensor(self.env.get_avail_actions())),  self.batch_size))
                    actions_ones = np.eye(self.env._num_actions)[actions]
                    last_action = actions
                    actions = [actions]
                    reward, terminated = self.env.step(np.array(actions[0]).squeeze().swapaxes(0,1), [self.iter_over for _ in range(self.batch_size)])  # step
                    reward = reward - 0.0001 if self.punish_step else reward
                    masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in terminated])
                    masks = masks.repeat(1,self.n_agents).unsqueeze(2)
                    bad_masks = masks
                    # share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None)
                    obs = np.array([self.env.get_obs()]).squeeze(0).swapaxes(0,1)

                    if self.use_centralized_V:
                        if self.shared:
                            share_obs = obs.reshape(self.batch_size, -1)
                        else:
                            share_obs = obs.reshape(self.batch_size, -1)
                        share_obs = np.expand_dims(share_obs, 1).repeat(self.n_agents, axis=1)
                    else:
                        share_obs = obs
                    data = share_obs.squeeze(), obs.squeeze(), rnn_state.squeeze(1), rnn_state_critic.squeeze(2), actions_ones.squeeze(), action_log_prob.squeeze(1), value, reward, masks, bad_masks, None, available_actions.squeeze()

                    if not test_mode:
                        share_obs = self.insert(data, iter_over=self.iter_over)  # insert


            else:
                if self.args.env_args["name"] == "iw2si":
                    keys_carried = [deepcopy(el) for el in self.env.get_carryings()]
                    keys_carried = [keys_carried[kl] for kl in range(len(keys_carried))]


                    for i in range(self.n_agents):
                        l = 0
                        for j in range(self.batch_size):

                            if keys_carried[j][i] is not None and last_action is not None and last_action[:][j][i] == 3 and had_key[j][i] == True:
                                self.statistics_dict["key_pickup"][i][j] += 1
                                l += 1

                                if sum(np.array(self.statistics_dict["key_first"])[:,j]) == 0: # between all agents in a batch
                                    self.statistics_dict["key_first"][i][j] = 1




                if self.args.name in [ "coma", "maven", "pg"]:



                    self.batch.update(pre_transition_data, ts=self.t)
                    actions = self.mac.select_actions(self.batch,
                                                  t_ep=self.t,
                                                  t_env=self.t_env,
                                                  test_mode=test_mode,
                                                  )
                  #  print("actions: "+str(actions), flush=True)
                    last_action = actions

                    if self.args.env_args["name"] == "iw2si":

                        for i in range(self.n_agents):
                            l = 0
                            for j in range(self.batch_size):

                                if keys_carried[j][i] is not None and actions[:][j][i] == 4:
                                    self.statistics_dict["key_dropped"][i][j] += 1
                                    l+=1
                        reward, terminated = self.env.step(actions, pre_transition_data["order"])
                        reward = reward - 0.0001 if self.punish_step else reward
                        # self.statistics_dict["key_dropped"][i] =  self.statistics_dict["key_dropped"][i]/l if l != 0 else self.statistics_dict["key_dropped"][i]
                        had_key = [[False if x is not None else True for x in l1] for l1 in keys_carried]
                        reward = reward[:, self.iter_over]

                    else:
                        reward, terminated = self.env.step(actions, [i for i in range(self.batch_size)])
                        reward = reward - 0.0001 if self.punish_step else reward



                elif self.args.name in["saf"]:
                    act_masks = np.array(self.env.get_avail_actions())

                    with torch.no_grad():
                        action, logprob, value,  = self.mac.get_actions(obs.swapaxes(0,1), state, act_masks, None,
                                                                                        obs_old.swapaxes(0,1))
                    last_action = action


                    reward, done = self.env.step(action,  [self.iter_over for _ in range(self.batch_size)])
                    reward = reward - 0.0001 if self.punish_step else reward
                    had_key = [[False if x is not None else True for x in l1] for l1 in keys_carried]
                    next_obs,  next_act_masks, = torch.from_numpy(np.array(self.env.get_obs()).squeeze()), torch.from_numpy(np.array(self.env.get_avail_actions()).squeeze())
                    next_state = torch.from_numpy(np.array(self.env.get_state()).squeeze())

                    if self.args.env_args["name"] == "iw2si":
                        for i in self.iter_over:
                            l = 0
                            for j in range(self.batch_size):

                                if keys_carried[j][i] is not None and action[j][0] == 4:
                                    self.statistics_dict["key_dropped"][i][j] += 1
                                    l += 1

                            #self.statistics_dict["key_dropped"][i] =  self.statistics_dict["key_dropped"][i]/l if l != 0 else self.statistics_dict["key_dropped"][i]

                    if not test_mode and len(self.mac.policy.obs_shape) == 3 and np.any(done == False):
                        if self.args.env_args["completion_signal"]:
                            mid_shape = int(obs.shape[2] / 3)
                            mid_shape_1 = int((mid_shape / 2))

                            obs = obs.swapaxes(0, 1).reshape(obs.shape[1],
                                                             obs.shape[0],
                                                             int(mid_shape / mid_shape_1),
                                                             mid_shape_1, 3)
                        else:
                            obs = obs.swapaxes(0,1).reshape(obs.shape[1], obs.shape[0], 3,3,3)
                        self.buffer.insert(
                            obs,
                            torch.from_numpy(act_masks),
                            action,
                            logprob,
                            torch.from_numpy(reward),
                            value,
                            next_done,
                            step,)

                    elif not test_mode and np.any(done == False):
                        self.buffer.insert(
                            obs,
                            state,
                            act_masks,
                            action,
                            logprob,
                            reward,
                            value,
                            next_done,
                            step)

                    obs = next_obs
                    state = next_state
                    act_masks = next_act_masks
                    next_done = torch.from_numpy(done).unsqueeze(1).repeat(1,2)
                    terminated = done

                    end_batch = [next_obs, obs_old.swapaxes(0,1), next_done]

                else: #qmixers

                    available_actions = self.env.get_avail_actions()

                    if self.args.env_args["name"] == "iw2si":
                        had_key = [[False if x is not None else True for x in l1] for l1 in keys_carried]

                    if self.mac.last_action_obs:
                        ah = []
                        ah.append(torch.from_numpy(obs))
                        if last_action is None:
                            last_action = torch.zeros_like(torch.Tensor(available_actions)).swapaxes(0,1)
                            if len(obs.shape) == 2:
                                last_action = last_action.squeeze()
                            ah.append(last_action)
                        else:
                            cat_acts = torch.eye(self.env._num_actions)[last_action]
                            if len(cat_acts.shape) == 2:
                                cat_acts = cat_acts.unsqueeze(1)
                                ah.append(cat_acts)
                            else:
                                ah.append(cat_acts.swapaxes(0,1))

                        if len(ah[0].shape) == 2:
                            new_obs = torch.cat(ah, dim=1)
                        else:
                            new_obs = torch.cat(ah, dim=2)

                        actions = self.mac.sample(obs=new_obs, available_actions=available_actions)
                    else:
                        actions = self.mac.sample(obs=obs, available_actions=available_actions)


                    last_action = actions if type(actions) is not list else actions[0]


                    actions = actions[0] if type(actions) is list else actions
                    reward, terminated = self.env.step(actions, [self.iter_over for _ in range(self.batch_size)])
                    reward = reward - 0.0001 if self.punish_step else reward


                    next_obs = np.array([self.env.get_obs()]).reshape(self.batch_size, self.n_agents, -1)


                    done = terminated


                    if self.args.env_args["name"] == 'iw2si':
                        if self.args.mixer in[ "vdn", "qmix", "qtrans", "qatten"]:

                            obs = np.expand_dims(obs.squeeze(), 0) if self.batch_size == 1 else obs

                            transitions = {
                                'obs': obs if obs.shape[0] == self.batch_size else obs.swapaxes(0,1) ,
                                # 'state': state,
                                'actions': actions,
                                'available_actions': np.array(available_actions),
                                'rewards': reward,
                                'dones': done,
                                'filled': False,
                            }
                        else:
                            obs = np.expand_dims(obs,1)
                            obs = obs.reshape(self.batch_size,self.n_agents,-1)

                            transitions = {
                                'obs': obs, #if self.n_agents == 1 or  obs.shape[0] == self.batch_size else obs.swapaxes(0, 1),
                                # 'state': state,
                                'actions': actions,
                                'available_actions': np.array(available_actions),
                                'rewards': reward,
                                'dones': done,
                                'filled': False,
                            }

                        for i in self.iter_over:
                            l = 0
                            for j in range(self.batch_size):

                                if keys_carried[j][i] is not None and  actions[j][i] == 4:
                                    self.statistics_dict["key_dropped"][i][j] += 1
                                    l+=1

                           # self.statistics_dict["key_dropped"][i] =  self.statistics_dict["key_dropped"][i]/l if l != 0 else self.statistics_dict["key_dropped"][i]

                    else:

                        transitions = {
                            'obs': obs,
                            # 'state': state,
                            'actions': actions.squeeze(),
                            'available_actions': np.array(available_actions),
                            'rewards': np.expand_dims(reward,1),
                            'dones': done.squeeze(),
                            'filled': False,
                        }




                    self.buffer.store_transitions(transitions)

                    obs = next_obs.swapaxes(0,1)




            if not test_mode and self.args.env_args["name"]  == "iw2si":

                  if not self.env.puffer:
                    wtfaid = self.env._envs[0].grid.render(tile_size=32,
                                              agent_posits=self.env._envs[0].agent_posits,
                                              agent_dirs=self.env._envs[0].agent_dirs,
                                              extent=self.env._envs[0].agent_view_size,
                                              highlight_mask=[])

                  else:

                      wtfaid = self.env._envs[0].render_snap(tile_size=32,
                                                             agent_posits=self.env._envs[0].agent_posits(),
                                                             agent_dirs=self.env._envs[0].agent_dirs(),
                                                             extent=self.env._envs[0].agent_view_size(),
                                                             highlight_mask=[])



                  if not os.path.isdir('frames'):
                      os.mkdir('frames')

                  path = "frames/"

                  cv2.imwrite(os.path.join(path,'test_' + str(self.t) + '_'+ str(self.args.seed) +  '.png'), wtfaid[:, :, :])

            if self.args.env_args["name"] == "iw2si" and self.n_agents > 1:
                self.statistics_dict["agent_distance"].append(sum(self.env.euc_dist())/self.batch_size)


            if self.args.env_args["name"]  != "starcraft":
                for agent in self.iter_over:
                    step_agent = 0
                    success_agent = 0
                    key_agent = 0
                    drop_b_agent = 0
                    drop_a_agent = 0
                    self.statistics_dict["reward"][agent].append(reward[:,agent])

                    for batch in range(self.batch_size):


                            if 0.5 in reward[batch]:
                                step_agent += self.t
                                if self.statistics_dict["drop_before"][agent][batch] == 0:
                                    self.statistics_dict["drop_before"][agent][batch] = self.statistics_dict["key_dropped"][agent][batch]

                            if reward[batch][agent] > 1:
                                success_agent += self.t
                            if self.statistics_dict["drop_before"][agent][batch] > 0 and terminated[batch]: # if door opened and episode is over

                                self.statistics_dict["drop_after"][agent][batch] = self.statistics_dict["key_dropped"][agent][batch] - self.statistics_dict["drop_before"][agent][batch]

                            if  self.args.env_args["name"]  == "iw2si" and self.env.get_carryings()[batch][agent] is not None:
                                key_agent += 1
                            if type(last_action) is list:
                                last_action = np.array(last_action).squeeze().swapaxes(0,1)

                            if reward[batch, agent] == 0.5 and reward[batch, agent] < 0.5+0.5*self.n_agents  and last_action is not None and last_action[:, self.iter_over][batch, agent] == 5 and self.statistics_dict["door_first"][agent][ batch] == 0:
                                self.statistics_dict["door_first"][agent][batch] = 1



                    if step_agent > 0:
                        self.statistics_dict["reward_at_step"][agent].append( step_agent)

                    if  self.args.env_args["name"]  == "iw2si" and key_agent > 0:
                        self.statistics_dict["holds_key"][agent].append(key_agent)
                    if success_agent > 0:
                        self.statistics_dict["success_at_step"][0].append(success_agent/self.args.n_agents)


            else:
                self.statistics_dict["reward"].append(reward[:])


            episode_return += reward



            if not self.ppo:
                if  self.args.name in [ "coma", "maven", "pg"]:

                    if self.args.env_args["name"] != "starcraft":
                        reward = reward[:,self.iter_over]
                        post_transition_data = {
                        "actions": actions[:,self.iter_over],
                        "reward": [(reward,)],
                        "terminated": [(terminated ),
                                       ],}


                        self.batch.update(post_transition_data, ts=self.t)
                    else:

                        post_transition_data = {
                            "actions": actions,
                            "reward": [(reward,)],
                            "terminated": [(terminated, ),
                                           ], }

                        self.batch.update(post_transition_data, ts=self.t)


            self.t += 1
            self.episode_step += 1

            self.statistics_dict["total_steps"] = self.t


            step += 1


        keys_carried = np.array(self.env.get_carryings())[:,self.iter_over]

        if not self.ppo: #COMA, MAVEN, PG, QMIX


            if  self.args.name in [ "coma", "maven", "pg"]:

                if self.args.env_args["name"] != "starcraft":
                    last_data = {
                        "state": self.env.get_state(),
                        "avail_actions": np.array(self.env.get_avail_actions())[:,self.iter_over],
                        "obs": np.array(self.env.get_obs())[self.iter_over,:],

                    }

                    if type(self.mac.hidden_states) is not list:
                        last_data["hiddens"] = self.mac.hidden_states.reshape(self.batch_size,
                                                                          self.n_agents,
                                                                          -1)[:,self.iter_over]

                    else:
                        last_data["hiddens"] = np.stack([x.reshape(self.batch_size,
                                                                          1,
                                                                          -1).detach().numpy() for x in self.mac.hidden_states]).swapaxes(0,1).swapaxes(1,2)

                else:
                    last_data = {
                        "state": np.array([self.env.get_state()]).swapaxes(0, 1),
                        "avail_actions": np.array([self.env.get_avail_actions()]).swapaxes(0, 1).swapaxes(2, 3),
                        "obs": np.expand_dims(obs, 1),
                    }

                    if type(self.mac.hidden_states) is not list:
                        last_data["hiddens"] = self.mac.hidden_states.reshape(self.batch_size,
                                                                              self.n_agents,
                                                                              -1)[:, self.iter_over]

                    else:
                        last_data["hiddens"] = np.stack([x.reshape(self.batch_size,
                                                          1,
                                                          -1).detach().numpy() for x in self.mac.hidden_states]).swapaxes(0,1).swapaxes(1,2)



              #  i
                self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

                self.batch.update({"actions": actions[:,self.iter_over]}, ts=self.t)





        elif self.shared and self.env.name != "forced_coordination" and self.alg_name != "random":

            aval = np.array(np.split(_t2n(torch.Tensor(self.env.get_avail_actions())), self.batch_size)).squeeze(1)
            aval = aval.reshape((aval.shape[0] * aval.shape[1], aval.shape[2]))

            new_obs = None

            if self.mac.policy.last_action_obs:
                new_obs = []
                ah = []
                obs = self.buffer.share_obs[step]

                if step == 0:
                    acts = np.zeros_like(self.buffer.actions[step])

                else:
                    acts = self.buffer.actions[step - 1]

                ah.append(torch.from_numpy(obs))
                ah.append(torch.from_numpy(acts))
                new_obs = torch.cat(ah, dim=2)

            value, actions, action_log_prob, rnn_state, rnn_state_critic = self.mac.policy.select_actions(
                    cent_obs = np.concatenate(self.buffer.share_obs[step]),
                    obs=self.buffer.obs[step],
                    rnn_states_actor=self.buffer.rnn_states[step],
                    rnn_states_critic= np.concatenate(self.buffer.rnn_states_critic[step]),
                    masks=self.buffer.masks[step],
                    available_actions =aval,
                    deterministic  =test_mode,
                    last_action = new_obs)

            value = value.reshape(self.batch_size, self.n_agents, 1, -1)
            actions = actions.reshape(self.batch_size, self.n_agents, 1, -1)
            action_log_prob = action_log_prob.reshape(self.batch_size, self.n_agents, 1, -1)
            rnn_state = rnn_state.reshape(self.batch_size, self.n_agents, 1, -1)
            rnn_state_critic = rnn_state_critic.reshape(self.batch_size, self.n_agents, 1, -1)


            if  self.args.env_args["name"]  == "iw2si":
                for i in self.iter_over:
                    l = 1
                    for j in range(self.batch_size):

                        if keys_carried[j][i] is not None and actions[j][i] == 4:
                            self.statistics_dict["key_dropped"][i][j] += 1
                            l += 1

                    #self.statistics_dict["key_dropped"][i] = self.statistics_dict["key_dropped"][i] / l if l != 0 else self.statistics_dict["key_dropped"][i]

            value = np.array(_t2n(value))
            action_log_prob = np.array(_t2n(action_log_prob))
            rnn_state = np.array(_t2n(rnn_state))
            rnn_state_critic = np.array(_t2n(rnn_state_critic))
            available_actions = np.array(np.split(_t2n(torch.Tensor(self.env.get_avail_actions())), self.batch_size))
            actions_ones = np.eye(self.env._num_actions)[actions.cpu()]

            actions = [actions]
            if  self.args.env_args["name"]  == "iw2si":
                agent_order = [self.iter_over for _ in range(self.batch_size)]
                reward, terminated = self.env.step(actions[0].squeeze(),agent_order)  # step
                reward = reward - 0.0001 if self.punish_step else reward
                obs = np.array([self.env.get_obs()]).squeeze(0).swapaxes(0, 1)


#
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in terminated])
            masks = masks.repeat(1, self.n_agents).unsqueeze(2).detach().cpu().numpy()
            bad_masks = masks


            if self.use_centralized_V:
                if self.shared:
                    share_obs = obs.reshape(self.batch_size, -1)
                else:
                    share_obs = obs.reshape(self.batch_size, -1)
                share_obs = np.expand_dims(share_obs, 1).repeat(self.n_agents, axis=1)
            else:
                share_obs = obs

            # share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None)


            if self.args.env_args["name"] != "starcraft":
                data = (share_obs, obs.squeeze() if self.n_agents > 1 else obs.squeeze(1), rnn_state, rnn_state_critic,
            actions_ones.squeeze() if self.n_agents >1 else actions_ones.squeeze(1).squeeze(1), action_log_prob.squeeze(2),
            value.squeeze(2), reward, masks, bad_masks, None, available_actions)

                if not test_mode:
                    share_obs = self.insert(data, iter_over=self.iter_over)


            if self.args.env_args["name"]  != "starcraft":
                for agent in self.iter_over:
                    step_agent = 0
                    success_agent = 0
                    key_agent = 0
                    self.statistics_dict["reward"][agent].append(reward[:,agent])
                    for batch in range(self.batch_size):


                            if reward[batch][agent] == 0.5 :
                                step_agent += self.t

                            if reward[batch][agent] >= 1:
                                   success_agent += self.t

                            if  self.args.env_args["name"]  == "iw2si" and self.env.get_carryings()[batch][agent] is not None:
                                key_agent += 1

                    if step_agent > 0:
                        self.statistics_dict["reward_at_step"][agent].append( step_agent)
                    if  self.args.env_args["name"]  == "iw2si" and key_agent > 0:
                        self.statistics_dict["holds_key"][agent].append(key_agent)
                    if success_agent > 0:
                        self.statistics_dict["success_at_step"][0].append(success_agent/self.args.n_agents)

            else:
                self.statistics_dict["reward"].append(reward[:])


        elif self.alg_name != "random":

            values = []
            actions = []
            action_log_probs = []
            rnn_states = []
            rnn_state_critics = []


            for i in self.iter_over:
                aval = np.array(
                    np.split(_t2n(torch.Tensor(self.env.get_avail_actions())), self.batch_size)).squeeze()[:, i,
                       :]

                value, action, action_log_prob, rnn_state, rnn_state_critic = self.mac.policy[i].select_actions(
                    cent_obs=self.buffer[i].share_obs[step],
                    obs=np.expand_dims(self.buffer[i].obs[step], axis=1),
                    rnn_states_actor=np.expand_dims(self.buffer[i].rnn_states[step], axis=1),
                    rnn_states_critic=self.buffer[i].rnn_states_critic[step],
                    masks=np.expand_dims(self.buffer[i].masks[step],axis=1),
                    available_actions = aval,
                    deterministic=test_mode
                )  # collect


                l = 1
                for j in range(self.batch_size):

                    if keys_carried[j][i] is not None and action[:,j][0] == 4:
                            self.statistics_dict["key_dropped"][i][j] += 1
                            l += 1

                #self.statistics_dict["key_dropped"][i] /= l

                values.append(value.detach().cpu().numpy())
                actions.append(np.array(_t2n(action)))
                action_log_probs.append(action_log_prob.detach().cpu().numpy())
                rnn_state = np.array(_t2n(torch.Tensor(rnn_state)))
                rnn_state_critic = np.array(
                    np.split(_t2n(torch.Tensor(rnn_state_critic)), self.batch_size))
                rnn_states.append(rnn_state)
                rnn_state_critics.append(rnn_state_critic)

            value = np.array(values)
            action_log_prob = np.array(action_log_probs)
            rnn_state = np.stack(rnn_states)
            rnn_state_critic = np.stack(rnn_state_critics)
            available_actions = np.array(np.split(_t2n(torch.Tensor(self.env.get_avail_actions())), self.batch_size))
            actions_ones = np.eye(self.env._num_actions)[actions]

            actions = [actions]
            reward, terminated = self.env.step(np.array(actions[0]).squeeze().swapaxes(0, 1), [self.iter_over for _ in range(self.batch_size)])  # step
            reward = reward - 0.0001 if self.punish_step else reward
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in terminated])
            masks = masks.repeat(1, self.n_agents).unsqueeze(2)
            bad_masks = masks
            obs = np.array([self.env.get_obs()]).squeeze(0).swapaxes(0, 1)
            if self.use_centralized_V:
                if self.shared:
                    share_obs = obs.reshape(self.batch_size, -1)
                else:
                    share_obs = obs.reshape(self.batch_size, -1)
                share_obs = np.expand_dims(share_obs, 1).repeat(self.n_agents, axis=1)
            else:
                share_obs = obs
            # share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None)

            data = share_obs.squeeze(), obs.squeeze(), rnn_state.squeeze(1), rnn_state_critic.squeeze(
                2), actions_ones.squeeze(), action_log_prob.squeeze(1), value, reward, masks, bad_masks, None, available_actions.squeeze()

            if not test_mode:
                share_obs = self.insert(data, iter_over=self.iter_over)  # insert


            for agent in self.iter_over:
                step_agent = 0
                success_agent = 0
                key_agent = 0
                self.statistics_dict["reward"][agent].append(reward[:, agent])

                for batch in range(self.batch_size):

                    if reward[batch][agent] == 0.5:
                        step_agent += self.t

                    if reward[batch][agent] >= 1 and not self.reward_coop:
                        success_agent += self.t

                    elif reward[batch][agent] > 1.5 and self.reward_coop:
                        success_agent += self.t



                    if self.env.get_carryings()[batch][agent] is not None:
                        key_agent += 1

                if step_agent > 0:
                    self.statistics_dict["reward_at_step"][agent].append(step_agent)
                if key_agent > 0:
                    self.statistics_dict["holds_key"][agent].append(key_agent)
                if success_agent > 0:
                    self.statistics_dict["success_at_step"][0].append(success_agent / self.args.n_agents)

        if self.args.name =="qmixer":
            for _ in range(self.t, self.episode_limit):
                self.buffer.episode_data.fill_mask()



        if self.args.env_args["name"] == "iw2si":
            succ_check = np.array(self.statistics_dict["reward"][0])
            succ_check = succ_check.sum(axis=0)
            if self.n_agents > 1:
                succ_check2 = np.array(self.statistics_dict["reward"][1])
                succ_check2 = succ_check2.sum(axis=0)

            if self.credit_easy_af and self.n_agents > 1:
                succ_check = np.array(self.statistics_dict["reward"][0])
                succ_check[succ_check < 0] = 0
                succ_check2 = np.array(self.statistics_dict["reward"][1])
                succ_check2[succ_check2 < 0] = 0
                succ_check = succ_check.sum(axis=0)
                succ_check2 = succ_check2.sum(axis=0)

            add_succ = 0.5*self.n_agents if self.reward_coop else 0


            for i in range(self.batch_size):

                if succ_check[i] >= 1.5+add_succ  : # all will share reward if success
                            self.statistics_dict["success"] += 1

                else:
                        self.statistics_dict["success"] += 0

            self.statistics_dict["success"]/=self.batch_size

        if self.args.env_args["name"] == "iw2si" :
            add_immediate = 0.5 if not self.only_sparse and self.n_agents > 1 else 0
            add_collective = sum([0.5] * self.n_agents)  if not self.only_immediate and self.n_agents > 1 else 0.5
            add_easy = 0.5*self.n_agents if self.reward_coop else 0

        else:
            add_collective = 1
            add_immediate = 0
            add_easy = 0

        if not test_mode:
            train_dic["training_episode"]= self.episode
            train_dic["training_success_per_episode"] = self.statistics_dict["success"]
            if self.args.env_args["name"] == "iw2si":
                train_dic["training_average_agent_distance_per_episode"] =  np.mean(self.statistics_dict["agent_distance"])
            train_dic["total_steps"] =  self.statistics_dict["total_steps"]



            for i in self.iter_over:


                if self.args.env_args["name"] == "iw2si":

                    train_dic["normalized_training_reward_for_agent_" + str(i + 1)] = np.mean(
                        np.array(self.statistics_dict["reward"])[i].sum(0)) / (
                                                                                                  add_collective + add_immediate + add_easy)
                    train_dic["training_holding_key_for_agent_" + str(i + 1)] = np.mean(self.statistics_dict["holds_key"][i])
                    train_dic["training_reward_at_step_for_agent_" + str(i + 1)] = np.mean(self.statistics_dict["reward_at_step"][i])
                    train_dic["training_success_at_step_for_agent_" + str(i + 1)] = np.mean(self.statistics_dict["success_at_step"][i])
                    train_dic["training_keys_dropped_for_agent_" + str(i+1)] = np.mean(self.statistics_dict["key_dropped"][i])
                    train_dic["training_keys_pickup_for_agent_" + str(i + 1)] = np.mean(self.statistics_dict["key_pickup"][i])
                    train_dic["training_first_to_key_for_agent_" + str(i + 1)] = np.mean(self.statistics_dict["key_first"][i])
                    train_dic["training_first_to_door_for_agent_" + str(i + 1)] = np.mean(self.statistics_dict["door_first"][i])
                    train_dic["training_keys_drop_before_for_agent_" + str(i + 1)] = np.mean(self.statistics_dict["drop_before"][i])
                    train_dic["training_keys_drop_after_for_agent_" + str(i + 1)] = np.mean(self.statistics_dict["drop_after"][i])
                else:
                    train_dic["normalized_training_reward_for_agent_" + str(i + 1)] = np.mean(np.sum(np.array(list(self.statistics_dict["reward"])), axis=0))

        else:
            train_dic["testing_episode"] = self.episode
            train_dic["testing_success_per_episode"] = self.statistics_dict["success"]
            if self.args.env_args["name"] == "iw2si":
                train_dic["testing_average_agent_distance_per_episode"] = np.mean(self.statistics_dict["agent_distance"])

            for i in self.iter_over:

                if self.args.env_args["name"] == "iw2si":
                    train_dic["normalized_testing_reward_for_agent_" + str(i + 1)] = np.mean(
                        np.array(self.statistics_dict["reward"])[i].sum(0)) / (
                                add_collective + add_immediate + add_easy)
                    train_dic["testing_holding_key_for_agent_" + str(i + 1)] = np.mean(self.statistics_dict["holds_key"][i])
                    train_dic["testing_reward_at_step_for_agent_" + str(i + 1)] = np.mean(self.statistics_dict["reward_at_step"][i])
                    train_dic["testing_success_at_step_for_agent_" + str(i + 1)] = np.mean( self.statistics_dict["success_at_step"][i])
                    train_dic["testing_keys_dropped_for_agent_" + str(i + 1)] = np.mean(self.statistics_dict["key_dropped"][i])
                else:
                    train_dic["normalized_testing_reward_for_agent_" + str(i + 1)] = np.mean(
                        np.sum(np.array(list(self.statistics_dict["reward"])), axis=0))

        #     cur_stats = self.test_stats if test_mode else self.train_stats
     #   cur_returns = self.test_returns if test_mode else self.train_returns
     #   log_prefix = "test_" if test_mode else ""
        # print(cur_stats)
        # print(env_info)
#        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
     #   cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        self.count += 1
        # print('n_episodes', cur_stats["n_episodes"], 'n_battle_won', cur_stats['battle_won'], 'test_mode', test_mode, 'count', self.count)
      #  cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += int(self.t)

      #  cur_returns.append(episode_return)

     #   if  self.args.name in [ "coma", "maven"] and test_mode and (len(self.test_returns) == self.args.test_nepisode):
    #        self._log(cur_returns, cur_stats, log_prefix)
    #    elif  self.args.name in [ "coma", "maven"] and self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
    #        self._log(cur_returns, cur_stats, log_prefix)
    #        if hasattr(self.mac.action_selector, "epsilon"):
    #            self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
    #        self.log_train_stats_t = self.t_env

        if  self.args.name in [ "coma", "maven", "pg"]:
            return self.batch, train_dic

        elif self.args.name in ["saf"]:
            return end_batch, train_dic

        else:
            return 1, train_dic

    def _log(self, returns, stats, prefix):
       # self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        #self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()
        # print('n_episodes', stats["n_episodes"], 'n_battle_won', stats['battle_won'])
        # print(stats['battle_won']/stats["n_episodes"])
    #    for k, v in stats.items():
        #    if k != "n_episodes":
              #  self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        #stats.clear()

    def insert(self, data, iter_over):

        if self.shared:
            # reorganize data by iter_over

            reorder = np.array(iter_over).argsort()
            data = list(data)
            for data_index in range(len(data)):
                if data[data_index] is not None and data[data_index].shape[1] == len(iter_over):
                    data[data_index] = data[data_index][:,reorder] #batch, agents, ....
                elif data[data_index] is not None and data[data_index].shape[2] == len(iter_over):
                    data[data_index] = data[data_index][:, :,reorder]

            share_obs, obs, rnn_states, rnn_state_critics, actions, action_log_prob, value, reward, masks, bad_masks, dead_agents, available_actions = data

            if self.args.env_args["name"] == "starcraft":
                share_obs = share_obs.squeeze()
            self.buffer.insert(share_obs=share_obs,
                          obs= obs,
                           rnn_states_actor=rnn_states,
                           rnn_states_critic=rnn_state_critics,
                           actions=actions,
                          action_log_probs= action_log_prob,
                           value_preds=value,
                          rewards= np.expand_dims(reward,2),
                          masks= masks,
                            available_actions=available_actions.squeeze())
            return share_obs

        else:
            for i in self.iter_over:
                share_obs, obs, rnn_states, rnn_state_critics, actions, action_log_prob, value, reward, masks, bad_masks, dead_agents, available_actions = data


                self.buffer[i].insert(share_obs=share_obs[:,i,:],
                               obs=obs[:,i,:],
                               rnn_states=rnn_states[i],
                               rnn_states_critic=rnn_state_critics[i],
                               actions=actions[i],
                               action_log_probs=action_log_prob[i],
                               value_preds=value[i],
                               rewards=np.expand_dims(reward[:,i],1),
                               masks=masks.detach().cpu().numpy()[:,i,:],
                               available_actions = available_actions[:,i,:]   )

            return data[0]
