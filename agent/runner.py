import os
import time
import argparse
import json
import torch.nn as nn
from utils import RemoteConnection

import evaluation_pb2
import evaluation_pb2_grpc
import grpc
import gymnasium as gym

import stable_baselines3 as sb3
import sb3_contrib
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from pingpong_wrapper import PingPongWrapper



def load_policy(args, Agent):
    if args.env_header:
        exec(args.env_header)

    policy = args.agent_kwargs.pop("policy", None)
    policy = "MlpPolicy" if policy is None else policy

    if policy not in Agent.policy_aliases.keys():
        policy = eval(policy)

    return policy


def load_agent(args):
    # Get the agent
    if hasattr(sb3, args.agent) or hasattr(sb3_contrib, args.agent):
        # Use the sb3 agent
        Agent = getattr(sb3_contrib, args.agent, getattr(sb3, args.agent, None))
    else:
        # Use the custom agent
        if args.env_header:
            exec(args.env_header)

        Agent = eval(args.agent)

    return Agent


def process_variable(args):
    if args.env_header:
        exec(args.env_header)

    # Agent seed
    args.agent_kwargs["seed"] = args.seed

    # Policy kwargs
    if "policy_kwargs" in args.agent_kwargs:
        if "features_extractor_class" in args.agent_kwargs["policy_kwargs"]:
            args.agent_kwargs["policy_kwargs"]["features_extractor_class"] = eval(
                args.agent_kwargs["policy_kwargs"]["features_extractor_class"]
            )
        if "activation_fn" in args.agent_kwargs["policy_kwargs"]:
            args.agent_kwargs["policy_kwargs"]["activation_fn"] = eval(
                args.agent_kwargs["policy_kwargs"]["activation_fn"]
            )

    return args


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


def main():
    # TODO: 进行修改 Model and agent

    args = json.load(open("checkpoints/tabletennis_fk_ppo_p2.json"))
    args = argparse.Namespace(**args)

    Agent = load_agent(args=args)

    assert args.load_model_dir is not None, "Please specify the model to load"
    print(f"Loading model from {args.load_model_dir}")
    model = Agent.load(
        os.path.join(args.load_model_dir, "model.zip"),
        **args.load_kwargs if hasattr(args, "load_kwargs") else {},
    )
    

    LOCAL_EVALUATION = os.environ.get("LOCAL_EVALUATION")
    view = False

    if LOCAL_EVALUATION:
        rc = RemoteConnection("environment:8085")
    else:
        rc = RemoteConnection("localhost:8085")
    env = PingPongWrapper(
        rc, obs_keys=args.single_env_kwargs["obs_keys"], kp_scale=args.wrapper_list["PingPongWrapperP2"]["kp_scale"]
    )
    vec_norm = VecNormalize.load(os.path.join(args.load_model_dir, "env.zip"), DummyVecEnv([lambda: env]))

    flat_completed = None
    trial = 0

    if view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model.ptr, env.data.ptr)

    while not flat_completed:
        flag_trial = None  # this flag will detect the end of an episode/trial
        ret = 0

        print(f"PINGPONG: Start Resetting the environment and get 1st obs of iter {trial}")

        obs, info = env.reset()

        if view and viewer.is_running():
            viewer.sync()
            time.sleep(0.05)

        # print(f"Trial: {trial}, flat_completed: {flat_completed}")
        counter = 0
        rewards = 0
        while not flag_trial:
            # env.rc_obs_dict["ball_pos"]
            obs = vec_norm.normalize_obs(obs[None, :])
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, flag_trial, flat_completed = env.step(action[0])
            rewards += reward

            if view and viewer.is_running():
                viewer.sync()
                time.sleep(0.05)

            counter += 1
        
        print(f"Trail: {trial}, rewards: {rewards}")
        trial += 1

    if view and viewer.is_running():
        viewer.close()

if __name__ == "__main__":
    main()