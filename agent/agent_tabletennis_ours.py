import os
import pickle
import time

import copy
import numpy as np

import evaluation_pb2
import evaluation_pb2_grpc
import grpc
import gymnasium as gym
from stable_baselines3.common import policies
import torch as th

from utils import RemoteConnection

"""
Define your custom observation keys here
"""
custom_obs_keys = [ 
    "time",
    'pelvis_pos', 
    'body_qpos', 
    'body_qvel', 
    'ball_pos', 
    'ball_vel', 
    'paddle_pos', 
    'paddle_vel', 
    'paddle_ori', 
    'reach_err', 
    'touching_info', 
    'act',
]

def pack_for_grpc(entity):
    return pickle.dumps(entity)

def unpack_for_grpc(entity):
    return pickle.loads(entity)

class Policy:

    def __init__(self, env):
        self.action_space = env.action_space
        self.policy = policies.ActorCriticPolicy(observation_space=env.observation_space,
                                    action_space=env.action_space,
                                    lr_schedule=lambda _: th.finfo(th.float32).max,
                                    net_arch = [dict(pi=[1024, 1024, 512, 512, 256, 256], vf=[1024, 1024, 512, 512, 256, 256])],
                                    activation_fn=th.nn.SiLU,
                                    )
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "dagger_policy_5400000.zip")
        self.model = self.policy.load(model_path)

    def __call__(self, obs):
        action, _ = self.model.predict(obs)
        return action

def get_custom_observation(rc, obs_keys):
    """
    Use this function to create an observation vector from the 
    environment provided observation dict for your own policy.
    By using the same keys as in your local training, you can ensure that 
    your observation still works.
    """

    obs_dict = rc.get_obsdict()
    # add new features here that can be computed from obs_dict
    # obs_dict['qpos_without_xy'] = np.array(obs_dict['internal_qpos'][2:35].copy())

    return rc.obsdict2obsvec(obs_dict, obs_keys)


time.sleep(10)

LOCAL_EVALUATION = os.environ.get("LOCAL_EVALUATION")

if LOCAL_EVALUATION:
    rc = RemoteConnection("environment:8085")
else:
    rc = RemoteConnection("localhost:8085")

policy = Policy(rc)

# compute correct observation space using the custom keys
shape = get_custom_observation(rc, custom_obs_keys).shape
rc.set_output_keys(custom_obs_keys)

flat_completed = None
trial = 0
while not flat_completed:
    flag_trial = None # this flag will detect the end of an episode/trial
    ret = 0

    print(f"PINGPONG: Start Resetting the environment and get 1st obs of iter {trial}")
    
    obs, _ = rc.reset()

    print(f"Trial: {trial}, flat_completed: {flat_completed}")
    counter = 0
    while not flag_trial:

        ################################################
        ## Replace with your trained policy.
        action = policy(obs)
        ################################################

        base = rc.act_on_environment(action)

        obs = base["feedback"][0]
        flag_trial = base["feedback"][2]
        flat_completed = base["eval_completed"]
        ret += base["feedback"][1]

        # 判断球是否打到对面 TODO: 为什么这个episode结束的这么快？
        touch_info = obs[-279:-273]
        print(f"Touch info: {touch_info}")
        if touch_info[1] == 1:
            print("Ball hit the own")
        if touch_info[0] == 1:
            print("Ball hit the paddle")
        elif touch_info[2] == 1:
            print("Ball hit the opponent")

        if flag_trial:
            print(f"Return was {ret}")
            print("*" * 100)
            break
        counter += 1
    trial += 1
