import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
#from sacred import Experiment, SETTINGS
#from sacred.observers import FileStorageObserver
##from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
#from utils.logging import get_logger
import yaml
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))


from run import run


#SETTINGS['CAPTURE_MODE'] = "sys" # set to "no" if you want to see stdout/stderr in console
#logger = get_logger()
#th.set_num_threads(1)
#ex = Experiment("pymarl")
#ex.logger = logger
#ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


#@ex.main
def my_main( _config):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(np.int64(config["seed"]))
   # th.manual_seed(config["seed"])
    config['env_args']['seed'] = np.int64(config["seed"])

    # run the framework
    run( config,)


def _get_config(params, arg_name, subfolder,):

    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict

def update_dict(dict, cmd_line):
    for _i, _v in enumerate(cmd_line):
        if _v.split("=")[0] in list(dict.keys()) and _v.split("=")[0] != 'env_args.p':
            if _v.split("=")[1] in ['True', 'False']:
                lop = eval(_v.split("=")[1])
            else:
                lop = _v.split("=")[1]
            dict[_v.split("=")[0]] = lop

        elif _v.split("=")[0] == 'env_args.p':

            dict['env_args']['p'] = [float(val) for val in _v.split("=")[1].split(",")]

        elif _v.split("=")[0].split(".")[0] == "env_args":

            if _v.split("=")[1] in ['True', 'False']:
                lop = eval(_v.split("=")[1])
            else:
                lop = _v.split("=")[1]
            dict['env_args'][_v.split("=")[0].split(".")[1]] = lop



def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == '__main__':
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)
    update_dict(config_dict, sys.argv[4:])

    # now add all the config to sacred
    #ex.add_config(config_dict)
    my_main(config_dict)
    # Save to disk by default for sacred
    #logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
   # ex.observers.append(FileStorageObserver.create(file_obs_path))

   # ex.run_commandline(params)

