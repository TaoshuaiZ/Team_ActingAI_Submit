import time
import gymnasium as gym
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from myosuite.envs.myo.myochallenge.tabletennis_v0 import PingpongContactLabels, geom_id_to_label
from planner import *

class PingPongWrapper(gym.Wrapper):
    def __init__(self, rc, obs_keys, kp_scale=1):
        env = gym.make("myoChallengeTableTennisP2-v0", obs_keys=obs_keys)
        super().__init__(env)
        self.rc = rc
        self.obs_keys = obs_keys
        rc.set_output_keys(obs_keys)

        # Create alias
        self.model = self.env.sim.model
        self.data = self.env.sim.data

        self.kp_scale = kp_scale

        self.hit_time = None
        self.hit_tolerance = 0.05
        self.hit_pos = None
        self.hit_with_paddle = 0
        self.hit_with_paddle_count = 2
        self.opponent_center = np.array([-0.685, 0.04, 0.795])

        self.fk_data = mujoco.MjData(self.model.ptr)
        self.ball_data = mujoco.MjData(self.model.ptr)
        mujoco.mj_forward(self.model.ptr, self.ball_data)

        # if self.eval_mode:
        #     self.ball_data_viewer = mujoco.viewer.launch_passive(self.model.ptr, self.ball_data)

        self.equalities = []
        self.action_joints = []
        self.constrained_joints = []

        for i in range(self.model.neq):
            self.equalities.append([self.model.eq_obj1id[i], self.model.eq_obj2id[i], self.model.eq_data[i]])
            self.constrained_joints.append(self.model.eq_obj1id[i])

        for i in range(self.model.njnt):
            if (
                self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
                or self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_SLIDE
            ) and i not in self.constrained_joints:
                self.action_joints.append(i)

        # self.action_joints = self.action_joints[2:]  # remove pelvis_x and pelvis_y
        n_actions = len(self.action_joints)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.env.observation_space.shape[0] + 11,)
        )
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(n_actions,))
        self.action_low = self.model.jnt_range[:, 0][self.action_joints]
        self.action_high = self.model.jnt_range[:, 1][self.action_joints]
        self.action_to_qpos = self.model.jnt_qposadr[self.action_joints]

    def _target_length_to_activation(self, target_lengths):
        """
        Converts target lengths to activation levels via force computation.

        Args:
            target_length (np.ndarray): Target lengths of the actuators.

        Returns:
            np.ndarray: Activation levels for each actuator, clipped between 0 and 1.
        """
        activations = []
        for idx_actuator in range(self.model.nu):
            if self.model.actuator_dyntype[idx_actuator] != mujoco.mjtDyn.mjDYN_MUSCLE:
                continue

            length = self.data.actuator_length[idx_actuator]
            lengthrange = self.model.actuator_lengthrange[idx_actuator]
            velocity = self.data.actuator_velocity[idx_actuator]
            peak_force = self.model.actuator_biasprm[idx_actuator, 2]

            L_0 = (lengthrange[1] - lengthrange[0]) / (
                self.model.actuator_gainprm[idx_actuator, 1] - self.model.actuator_gainprm[idx_actuator, 0]
            )

            kp = self.kp_scale
            kd = 0.1 * kp
            force = (kp * (target_lengths[idx_actuator] - length) - kd * velocity) * peak_force / (lengthrange[1] - lengthrange[0])
            clipped_force = np.clip(force, -peak_force, 0)

            acc0 = self.model.actuator_acc0[idx_actuator]
            prmb = self.model.actuator_biasprm[idx_actuator, :9]
            prmg = self.model.actuator_gainprm[idx_actuator, :9]
            bias = mujoco.mju_muscleBias(length, lengthrange, acc0, prmb)
            gain = np.minimum(-1, mujoco.mju_muscleGain(length, velocity, lengthrange, acc0, prmg))
            activations.append(np.clip((clipped_force - bias) / gain, 0, 1))

        activations = np.stack(activations, axis=0)
        return np.clip(activations, 0, 1)

    def _get_target_actuator_length(self, qpos):
        for aid in range(self.action_space.shape[0]):
            self.fk_data.qpos[self.action_to_qpos[aid]] = qpos[aid]

        for id1, id2, eq_data in self.equalities:
            qpos_id1 = self.model.jnt_qposadr[id1]
            qpos_id2 = self.model.jnt_qposadr[id2]
            if id2 == -1:
                self.fk_data.qpos[qpos_id1] = self.model.qpos0[qpos_id1] + eq_data[0]
            else:
                self.fk_data.qpos[qpos_id1] = (
                    self.model.qpos0[qpos_id1]
                    + eq_data[0]
                    + eq_data[1] * (self.fk_data.qpos[qpos_id2] - self.model.qpos0[qpos_id2])
                    + eq_data[2] * (self.fk_data.qpos[qpos_id2] - self.model.qpos0[qpos_id2]) ** 2
                    + eq_data[3] * (self.fk_data.qpos[qpos_id2] - self.model.qpos0[qpos_id2]) ** 3
                    + eq_data[4] * (self.fk_data.qpos[qpos_id2] - self.model.qpos0[qpos_id2]) ** 4
                )
                self.fk_data.qpos[qpos_id1] = np.clip(
                    self.fk_data.qpos[qpos_id1], self.model.jnt_range[id1, 0], self.model.jnt_range[id1, 1]
                )

        mujoco.mj_forward(self.model.ptr, self.fk_data)

        return self.fk_data.actuator_length

    def reset(self, **kwargs):
        super().reset(**kwargs)

        obs, info = self.rc.reset()
        self.paddle_face_dir_local = np.array([0, 0, -1])

        # TODO： 进行修改command从obs中进行计算
        p_paddle, v_paddle, o_paddle, t_hit = self.get_high_command(self.obs_dict["ball_pos"].copy(), self.obs_dict["ball_vel"].copy())
        self.hit_pos_paddle = p_paddle
        self.v_paddle = v_paddle
        self.o_paddle = o_paddle
        self.hit_time = t_hit

        obs = np.concatenate([obs, p_paddle, v_paddle, o_paddle, np.array([self.hit_time])])

        return obs, info

    def get_high_command(self, cur_pos, cur_vel):
        """
        get high level command: p_paddle, v_paddle, t_hit, p_base
        return: 
            p_paddle: predicted paddle position
            v_paddle: predicted paddle velocity
            t_hit: time to hit the ball
            
        """
        # get hit pos
        hit_pos, v_i, t_hit = compute_hit_pos(cur_pos, cur_vel, hit_plane_x=1.8)
        # get out ball vel
        v_o = compute_out_vel(target_pos=self.opponent_center, hit_pos=hit_pos)

        # get v_paddle
        v_paddle = compute_racket_vel(v_i, v_o)

        # get paddle orientation TODO: 拍子朝向可能与拍子速度方向不一致
        o_paddle = vec_to_quat(self.paddle_face_dir_local, v_paddle / np.linalg.norm(v_paddle))

        # get p_pad
        rot = R.from_quat([o_paddle[1], o_paddle[2], o_paddle[3], o_paddle[0]])
        rot_mat = rot.as_matrix()
        pad_loacl = np.array([-0.07, 0, 0])
        pad_world = rot_mat.reshape(3, 3) @ pad_loacl
        p_paddle = hit_pos - pad_world


        return p_paddle, v_paddle, o_paddle, t_hit

    def step(self, action):
        assert len(action) == self.action_space.shape[0]
        action = np.clip(action, -1, 1)
        action = (action + 1) * (self.action_high - self.action_low) / 2 + self.action_low

        target_length = self._get_target_actuator_length(action)
        activations = self._target_length_to_activation(target_length)
        if self.env.normalize_act:
            # inverse the normalization
            activations = np.clip(activations, 1.0 / (1.0 + np.exp(7.5)), 1.0 / (1.0 + np.exp(-2.5)))
            input_actions = -0.2 * np.log(1.0 / activations - 1.0) + 0.5
            input_actions = np.clip(input_actions, -1, 1)
        else:
            input_actions = activations

        env_actions = np.concatenate([input_actions, action[:2]])
        _obs, _, _terminated, _truncated, _info = super().step(env_actions)
        ret = self.rc.act_on_environment(env_actions)
        obs = ret["feedback"][0]
        reward = ret["feedback"][1]
        flag_trial = ret["feedback"][2]
        flat_completed = ret["eval_completed"]

        obs = np.concatenate([obs, self.hit_pos_paddle, self.v_paddle, self.o_paddle, np.array([self.hit_time])])

        return obs, reward, flag_trial, flat_completed


def close(a, b):
    return np.abs(a - b).max() < 1e-3