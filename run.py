
import os
import pprint
import threading
import torch
import torch as th
import random
from types import SimpleNamespace as SN
from os.path import dirname, abspath
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from components.episode_buffer import SeparatedReplayBuffer, SharedReplayBuffer
from components.EpisodeData import ReplayBuffer as QmixBuffer
from collections import defaultdict
import numpy as np
import wandb
import datetime


def run( _config, ):

    # check args sanity


    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers



    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)


    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token

    if args.use_tensorboard:
        tb_logs_direc = os.path.join( dirname(dirname(abspath(__file__))), args.local_results_path,  "tb_logs")
        tb_exp_direc = os.path.join( tb_logs_direc, "{}").format(unique_token)


    # sacred is on by default


    # Run and train
    run_sequential(args=args)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

   # print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)



def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args):

    args.seed = int(args.seed)

    if args.random_order:

        iter_over = list(range(int(args.env_args["n_agents"])))
        random.shuffle(iter_over)

    elif args.turn_based:
        iter_over = list(range(int(args.env_args["n_agents"])))

    else:
        iter_over = list(range(int(args.env_args["n_agents"])))

    # Init runner so we can get env info
    args.env_args["n_agents"] = int( args.env_args["n_agents"])




    runner =  r_REGISTRY[args.runner](args=args,

                                                                                          logger=None,
                                                                                          agent_ids = [x for x in range(int(args.env_args["n_agents"]))],
                                                                                          iter_over=iter_over)

    if args.test_algs:
        test_runner = r_REGISTRY[args.runner](args=args,
                                          logger=None,
                                          agent_ids=[x for x in
                                                     range(int(
                                                         args.env_args[
                                                             "n_agents"]))],
                                          iter_over=iter_over)

    # Set up schemes and groups here


    share_proto = None
    share_enc = None



    if args.env_args["name"] == "iw2si":
        env_info = runner.get_env_info()

        if not args.puffer:
            args.n_agents = env_info["n_agents"][0]
            args.n_actions = env_info["n_actions"][0]
            args.state_shape = env_info["state_shape"][0]
            args.reward_shape = env_info["n_agents"][0]
            args.done_shape = env_info["n_agents"][0]
            args.obs_shape = env_info["obs_shape"][0].item() if not args.env_args["completion_signal"] else env_info["obs_shape"][0].item() +3
            if  args.env_args["key_signal"]:
                args.obs_shape = args.obs_shape + +3
            args.episode_limit = int(env_info["episode_limit"][0])

        else:
            args.n_agents = env_info["n_agents"]
            args.n_actions = env_info["n_actions"]
            args.state_shape = env_info["state_shape"]
            args.obs_shape = env_info["obs_shape"]

            args.reward_shape = env_info["n_agents"]
            args.done_shape = env_info["n_agents"]

            args.episode_limit = env_info["episode_limit"]

        args.batch_size = int(args.batch_size)
        args.buffer_size = int(args.buffer_size)
        args.batch_size_run = int(args.batch_size_run)





    else:
        args.n_agents = runner.env._n_agents
        args.n_actions = runner.env._num_actions
        args.state_shape = runner.env._observation_shape
        args.obs_shape = runner.env._observation_shape
        args.episode_limit = int(runner.env._episode_length)
        args.batch_size = int(args.batch_size)
        args.buffer_size = int(args.buffer_size)
        args.batch_size_run = int(args.batch_size_run)

    # Default/Base scheme
    if args.name in ["coma", "maven", "pg"]:

        if not args.puffer:
            scheme = {
                "state": {"vshape": env_info["state_shape"][0].item(), "dtype": th.int},
                "obs": {"vshape": args.obs_shape, "group": "agents"},
                "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
                "avail_actions": {"vshape": (env_info["n_actions"][0].item(),), "group": "agents", "dtype": th.int},
                "reward": {"vshape": int(env_info["reward_shape"][0].mean().item())},
                "terminated": {"vshape": (1,), "dtype": th.uint8},
                "hiddens": {"vshape": (args.rnn_hidden_dim,), "group": "agents", "dtype": th.long},
                "order": {"vshape": (1,), "group": "agents", },
                "noise": {"vshape": (args.noise_dim,), "dtype": th.int}
            }

        else:
            scheme = {
            "state": {"vshape": args.state_shape, "dtype": th.int},
            "obs": {"vshape": args.obs_shape , "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (args.n_actions,), "group": "agents", "dtype": th.int},
            "reward": {"vshape": int(args.reward_shape)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
            "hiddens": {"vshape": (args.rnn_hidden_dim,), "group": "agents", "dtype": th.long},
            "order": {"vshape": (1,), "group": "agents", },
            "noise": {"vshape": (args.noise_dim,), "dtype": th.int}
            }


        groups = {
            "agents": args.n_agents
        }
        preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
        }
    else:

        scheme = {
        "state": {"vshape": args.state_shape, "dtype": int},
        "obs": {"vshape": args.obs_shape, "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (args.n_actions,), "group": "agents", "dtype": int},
        "reward": {"vshape":  args.reward_shape, "dtype": th.long},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "hiddens": {"vshape": (args.critic_hidden_dim,), "group": "agents", "dtype": th.long},
        "noise": {"vshape": (args.noise_dim,), "dtype": th.int}

        }

        groups = {
            "agents": args.n_agents
        }
        preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
        }

    if args.name == "mappo" and args.share_buffer:


        s_obs = args.obs_shape*args.n_agents
        buffer = SharedReplayBuffer(args,
                                    num_agents= args.n_agents,
                                    obs_space=args.obs_shape,
                                    cent_obs_space=s_obs,
                                    act_space= args.n_actions)




    elif args.name == "ippo":
        buffer = []



        for i in range( args.n_agents):

            if args.use_centralized_V:
                s_obs = args.obs_shape*args.n_agents
            else:
                s_obs = args.obs_shape

            buffer.append(SeparatedReplayBuffer(args,
                                       obs_space=args.obs_shape,
                                    share_obs_space=s_obs,
                                   act_space= args.n_actions))


    elif args.name == "qmixer":
        buffer = QmixBuffer(args=args)



    elif args.name != "saf":

        buffer = ReplayBuffer(scheme, groups, args.buffer_size,   int(args.episode_limit) + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here


    rr_buffer, rr = None, None





    if args.weightsb:
        wandb.init(project='MarlCredit',
               config=args,
              name=args.run_name + str(datetime.datetime.now()))




    # Learner

    if args.name == "random":
        learner = le_REGISTRY[args.learner]( num_agents=args.n_agents )
        runner.setup(scheme=None, groups=None, preprocess=None, mac=learner, buffer=None)


    elif args.name == "mappo" and args.share_buffer:
        learner = le_REGISTRY[args.learner]( logger=None, args =args,)
        runner.setup(scheme=None, groups=None, preprocess=None, mac=learner, buffer=buffer)
        if args.test_algs:
            test_runner.setup(scheme=None, groups=None, preprocess=None, mac=learner, buffer=buffer)

       # if args.use_cuda:
         #   learner.cuda()

       # mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args, learner=None)
    elif args.name == "ippo" :
        #mixer = None if args.mixer == '' else mix_REGISTRY[args.mixer](args = args, cent_obs_dim =args.obs_shape*4 ,num_agents=args.n_agents, device=args.device)
        learner = (le_REGISTRY[args.learner](logger=None, args=args ,mixer= None))
        if args.test_algs:
            test_runner.setup(scheme=None, groups=None, preprocess=None, mac=learner, buffer=buffer)
        runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=learner, buffer=buffer)

    elif args.name == "qmixer":
        learner = (le_REGISTRY[args.learner]( args=args))
        runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=learner, buffer=buffer)
        if args.test_algs:
            test_runner.setup(scheme=None, groups=None, preprocess=None, mac=learner, buffer=buffer)

    elif args.name != "saf":
        mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args, learner=None)
        runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac, buffer=buffer)
        learner = le_REGISTRY[args.learner](mac=mac, scheme=buffer.scheme, logger=None, args=args)
        if args.test_algs:
            test_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=learner, buffer=buffer)

        if args.use_cuda:
            learner.cuda()

    else:
        if args.iid_agents:

            learner = [le_REGISTRY[args.learner]( args=args) for _ in range(args.n_agents)]

        learner = le_REGISTRY[args.learner]( args=args) # args, observation_space, action_space, state_space)
        runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=learner, buffer=learner.buffer)
        if args.test_algs:
            test_runner.setup(scheme=None, groups=None, preprocess=None, mac=learner, buffer=buffer)


    if args.load_model:

        model_path = os.path.join(args.local_results_path,
                                 args.name,
                                 args.env_args["name"],
                                 str(args.load_agentn),
                                 str(args.seed))


        learner.load_models(model_path, path2=None)

    # start training
    episode = 0
    count = 0
    #obs, _ = runner.env.reset() if runner.env.name == "iw2si" else (runner.env.reset(),1)
    while count < int(int(args.t_max)/args.episode_limit):

        # Run for a whole episode at a time

        if args.use_linear_lr_decay:
            learner.policy.lr_decay(count, int(int(args.t_max)/args.episode_limit))

        episode_batch, train_stats = runner.run(test_mode=False)
        count += 1

        if args.name == "mappo" and args.share_buffer:


            next_values = learner.policy.get_values(np.concatenate(runner.buffer.share_obs[-1]),
                                                         np.concatenate(runner.buffer.rnn_states_critic[-1]),
                                                         np.concatenate(runner.buffer.masks[-1]))
            next_values = np.array(np.split(_t2n(next_values), buffer.n_rollout_threads))
            infos = []
            j = False
            if type(rr) is list:
                j = True
            elif rr is not None:
                j = True

            if learner.la_update:

                indices_ent = runner.buffer.rewards.copy().reshape(args.n_agents, -1)
                indices_ent[indices_ent < 1] = 0
                indices_ent = torch.from_numpy(indices_ent)


            else:
                indices_ent = None

            if learner.entropy_method == 'max':
                obs = torch.from_numpy(runner.buffer.obs)[:-1]
                policy_entropies = learner._compute_policy_entropy(obs,
                                                                runner.buffer.available_actions[:-1],
                                                                runner.buffer.rnn_states[:-1],
                                                                runner.buffer.masks[:-1])

                policy_entropies  = policy_entropies.detach().numpy()
                runner.buffer.rewards -= learner._policy_ent_coeff * policy_entropies.reshape(runner.buffer.rewards.shape)
                train_stats["average_entropy"] = np.mean(policy_entropies)

            runner.buffer.compute_returns(next_values, learner.value_normalizer)

            info = learner.train(buffer= runner.buffer, indices=indices_ent)
            infos.append(info)

            runner.buffer.after_update()

            z = {**infos[0], **train_stats}

            if  args.test_algs:

                test_runner.mac = runner.mac
                test_runner.buffer = runner.buffer



            log_data(args.weightsb, args.write_to_scratch, z, args, count)


        elif args.name == "ippo": ## IPPO

            infos = []
            d = {}
            for i in iter_over: # compute
                next_value = learner.policy[i].get_values(buffer[i].share_obs[-1],
                                                        buffer[i].rnn_states_critic[-1],
                                                         buffer[i].masks[-1])
                next_value =  _t2n(next_value)
                runner.buffer[i].compute_returns(next_value, learner.value_normalizer[i])  # end of compute


            info = learner.train(buffer= buffer)
            infos.append(info)

            for i in iter_over:
                runner.buffer[i].after_update()

            z = {**infos[0], **train_stats}



            n_test_runs = max(1, int(args.test_nepisode) // int(runner.batch_size))
            if args.test_algs:

                test_runner.mac = runner.mac
                test_runner.buffer = runner.buffer


                test_infos = []
                test_runner.mac = runner.mac
                for _ in range(n_test_runs):
                   batch, test_info = runner.run(test_mode=True)
                   test_infos.append(test_info)

                tests = {}
                for dictionary in test_infos:
                    for key in dictionary:
                        if key in tests:
                           tests[key].append(dictionary[key])
                        else:
                            tests[key] = [dictionary[key]]

                for key in tests:
                    tests[key] = sum(tests[key]) / len(tests[key])

                z = {**tests, **z}

            log_data(args.weightsb, args.write_to_scratch, z, args, count)



        elif args.name == "random":
            runner.reset()

            log_data(args.weightsb, args.write_to_scratch, train_stats, args, count)


        else: #COMA, SAF, QMIXERs, SAF, PG
            d = {}
            if args.name == "saf":

                next_obs, obs_old, next_done = episode_batch

                with torch.no_grad():

                    bs = next_obs.shape[1]
                    n_ags = next_obs.shape[0]
                    next_obs = next_obs.reshape((-1,) + learner.policy.obs_shape)
                    next_obs = next_obs.to(torch.float32)
                    next_obs = learner.policy.conv(next_obs)
                    next_obs = next_obs.reshape(bs, n_ags, learner.policy.input_shape)
                    next_state = next_obs.reshape(bs, n_ags * learner.policy.input_shape)
                    next_state = next_state.unsqueeze(1).repeat(1, n_ags, 1)

                    if args.latent_kl:
                        next_obs_saf = learner.policy.SAF(next_obs)
                        next_z, _ = learner.policy.SAF.information_bottleneck(next_obs_saf, next_obs, obs_old)
                    else:
                        next_z = None

                    next_value = learner.policy.get_value(next_obs, next_state, next_z)

                    advantages, returns = learner.policy.compute_returns(learner.buffer, next_value, next_done.long())

                test_stats = learner.policy.train_step(learner.buffer, advantages, returns)

                train_stats = {**train_stats, **test_stats}

                if args.test_algs:

                    test_runner.mac = runner.mac
                    test_runner.buffer = runner.buffer



                log_data(args.weightsb, args.write_to_scratch, train_stats, args, count)


                if args.lr_decay:
                    learner.policy.update_lr(count, int(args.t_max/args.episode_limit))

            else: # COMA, DQN, MAVEN for Mixers

                if args.name == "qmixer":

                    # ReplayBuffer store the episode data
                    runner.buffer.store_episodes()
                    train_res_lst = []
                    if runner.buffer.size() > int(args.memory_warmup_size):
                        for _ in range(int(args.learner_update_freq)):
                            batch = runner.buffer.sample(int(args.batch_size)) # sample episodes
                            results = learner.learn(batch)
                            train_res_lst.append(results)

                        # Initialize a defaultdict to sum the values
                        sum_dict = defaultdict(int)

                        # Initialize a dictionary to count the occurrences of each key
                        count_dict = defaultdict(int)

                        for d in train_res_lst:
                            for key, value in d.items():
                                sum_dict[key] += value
                                count_dict[key] += 1

                        avg_dict = {key: sum_dict[key] / count_dict[key] for key in sum_dict}

                        z = {**avg_dict, **train_stats}

                    else:
                        z = {**train_stats}





                    if  args.test_algs :

                        test_infos = []
                        test_runner.mac = runner.mac
                        for _ in range(args.n_tests):
                            batch, test_info = test_runner.run(test_mode=True)
                            test_infos.append(test_info)

                        tests = {}
                        for dictionary in test_infos:
                            for key in dictionary:
                                if key in tests:
                                    tests[key].append(dictionary[key])
                                else:
                                    tests[ key] = [dictionary[key]]

                        for key in tests:
                            tests[key] = np.median(tests[key])

                        z = {**tests, **z}

                    if args.test_algs:

                        test_runner.mac = runner.mac
                        test_runner.buffer = runner.buffer



                    log_data(args.weightsb, args.write_to_scratch, z, args, count)

                else: # Maven and COMA & PG
                    runner.buffer.insert_episode_batch(episode_batch)

                    if  runner.buffer.can_sample(int(args.batch_size)):
                        episode_sample = runner.buffer.sample(int(args.batch_size))

                        # Truncate batch to only filled timesteps
                        max_ep_t = episode_sample.max_t_filled()
                        episode_sample = episode_sample[:, :max_ep_t]

                        if episode_sample.device != args.device:
                            episode_sample.to(args.device)

                        new_rewards, redist_steps = None, None


                        infos = []

                        #generate data from generators
                        pg_stats = learner.train(episode_sample, runner.t_env, episode, new_rewards, redist_steps, ) # pass data throught here
                        train_stats = {**pg_stats, **train_stats}


                    z = {**train_stats}

                    if args.test_algs:
                        test_runner.mac = runner.mac
                        test_runner.buffer = runner.buffer

                        last_test_T = runner.t_env
                        test_infos = []
                        test_runner.mac = runner.mac
                        for _ in range(int(args.n_tests)):
                            batch, test_info = test_runner.run(test_mode=True)
                            test_infos.append(test_info)

                        tests = {}
                        for dictionary in test_infos:
                            for key in dictionary:
                                if key in tests:
                                    tests[key].append(dictionary[key])
                                else:
                                    tests[key] = [dictionary[key]]

                        for key in tests:
                            tests[key] = sum(tests[key]) / len(tests[key])

                        z = {**tests, **train_stats}

                    if args.test_algs:

                        test_runner.mac = runner.mac
                        test_runner.buffer = runner.buffer



                    log_data(args.weightsb, args.write_to_scratch, z, args, count)



        if args.save_model and (episode+1) % int(args.save_freq) == 0:
            save_path = os.path.join(args.local_results_path,
                                     args.name,
                                     args.env_args["name"],
                                     str(args.n_agents),
                                     str(args.seed))

            os.makedirs(save_path, exist_ok=True)
            learner.save_models(save_path)

        episode+=1

        if args.random_order:

            random.shuffle(runner.iter_over)

        elif args.turn_based:
                iter_over = iter_over[-1:] + iter_over[:-1] # rotate elements to the right
                runner.iter_over = iter_over




        # Execute test runs once in a while

#    logger.dump_stats(os.path.join( args.local_results_path, "tb_logs", args.unique_token, "stats.pkl"))
    #runner.close_env()



def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config




def log_data(weights, write_txt, data, args ,count):
    if weights:
        wandb.log(data)

    if write_txt:

        file_name = os.path.join(args.data_dir + str(args.seed) +
                                 "_" + args.name + "_" + args.rr + "_" + args.mixer + "agents:" + str(
            args.n_agents) + ".txt")
        try:

            with open(file_name, 'a') as opened_file:

                vals = [*data.values()]

                if count == 1:
                    keys = [*data.keys()]
                    new_line = " ".join(str(val) for val in keys)
                    opened_file.write("%r\n" % new_line)

                new_line = " ".join(str(val) for val in vals)

                opened_file.write("%r\n" % new_line)
                opened_file.close()
        except:

            with open(file_name, 'w+') as opened_file:
                pass
            with open(file_name, 'a') as opened_file:
                vals = [*data.values()]

                if count == 1:
                    keys = [*data.keys()]
                    new_line = " ".join(str(val) for val in keys)
                    opened_file.write("%r\n" % new_line)

                new_line = " ".join(str(val) for val in vals)

                opened_file.write("%r\n" % new_line)
                opened_file.close()


def _t2n(x):
    return x.detach().cpu().numpy()