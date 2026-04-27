import time
import math
import cmath
import numpy as np
import torch as th
from stable_baselines3.common import policies
import mujoco
from scipy.spatial.transform import Rotation as R

def compute_land(cur_pos, cur_vel, h=0.806):
    """
    return: landing position (x, y, h), t_land, land_vel
    """
    g = -9.81

    a = 0.5 * g
    b = cur_vel[2]
    c = cur_pos[2] - h
    t = solve_quadratic(a,b,c)

    
    x = cur_pos[0] + cur_vel[0] * t
    y = cur_pos[1] + cur_vel[1] * t

    
    land_vel = np.array([cur_vel[0], cur_vel[1], cur_vel[2] + g * t])

    return np.array([x, y, h]), t, land_vel


def compute_land_after_hit(cur_pos, cur_vel, h=0.806):
    """
    return: landing position (x, y, h), t_land, land_vel
    """
    g = -9.81

    
    t_net = (0 - cur_pos[0]) / cur_vel[0]
    h_net = 0.5*g*t_net**2 + cur_vel[2]*t_net + cur_pos[2]
    if h_net < 0.965:
        return None, None, None

    
    a = 0.5 * g
    b = cur_vel[2]
    c = cur_pos[2] - h
    t = solve_quadratic(a,b,c)

    
    x = cur_pos[0] + cur_vel[0] * t
    y = cur_pos[1] + cur_vel[1] * t

    
    land_vel = np.array([cur_vel[0], cur_vel[1], cur_vel[2] + g * t])

    return np.array([x, y, h]), t, land_vel

def solve_quadratic(a, b, c):
    if a == 0:
        if b == 0:
            raise ValueError
        return -c / b
   
    delta = b**2 - 4*a*c
   
    if delta > 0:
        root1 = (-b + math.sqrt(delta)) / (2 * a)
        root2 = (-b - math.sqrt(delta)) / (2 * a)
        # print(root1, root2)
        return max(root1, root2)
    elif delta == 0:
        root = -b / (2 * a)
        # print(root)
        return root
    else:
        root1 = (-b + cmath.sqrt(delta)) / (2 * a)
        root2 = (-b - cmath.sqrt(delta)) / (2 * a)
        # print(root1,root2)
        return max(root1, root2)
    
def compute_hit_pos(cur_pos, cur_vel, hit_plane_x=1.8):

    g = -9.81
    land_pos, t_land, land_vel = compute_land(cur_pos, cur_vel)


    vel_bounce = -land_vel[2] 


    t_subhit = (hit_plane_x - land_pos[0]) / land_vel[0]
    t_subhit = np.round(t_land+t_subhit, 2) - t_land

    x_hit = t_subhit * land_vel[0] + land_pos[0]
    y_hit = land_pos[1] + land_vel[1] * t_subhit
    z_hit = 0.5 * g * t_subhit**2 + vel_bounce * t_subhit + land_pos[2]

    v_hit = np.array([land_vel[0], land_vel[1], vel_bounce+g*t_subhit])

    return np.array([x_hit, y_hit, z_hit]), v_hit, t_land+t_subhit


def compute_out_vel(target_pos, hit_pos, t_flight=0.29):
    g = np.array([0,0,-9.81])
    v_out = (target_pos - hit_pos) / t_flight - 0.5 * g * t_flight

    # v_out2 = (target_pos - hit_pos) / t_flight + 0.5 * 9.81 * t_flight
    # print(v_out2)
    return v_out


def compute_racket_vel(v_i, v_o, C_r=1):
    u = (v_o - v_i) / np.linalg.norm(v_o - v_i)

    v_racket = (v_o @ u + v_i @ u) / (1+C_r) * u 

    return v_racket

def vec_to_quat(from_vec, to_vec):
    from_vec = from_vec / np.linalg.norm(from_vec)
    to_vec = to_vec / np.linalg.norm(to_vec)

    axis = np.cross(from_vec, to_vec)
    angle = np.arccos(np.dot(from_vec, to_vec))

    if np.linalg.norm(axis) < 1e-6:
        if angle < 0.1:  
            return np.array([1.0, 0.0, 0.0, 0.0]) 
        else:  
            return np.array([0.0, 1.0, 0.0, 0.0])

    axis = axis / np.linalg.norm(axis)
    quat = np.concatenate(([math.cos(angle / 2)], axis * math.sin(angle / 2)))
    return quat




# if __name__ == "__main__":
    # device = th.device("cpu")

    # env = gym.make("myoChallengeTableTennisP1-v0")
    # obs, _ = env.reset()

    # obs = th.tensor(obs, dtype=th.float32, device=device)
    # collide = False
    # ref_face_dir = np.array([-1, 0, 0])
    # face_paddle = env.sim.data.site_xmat[env.id_info.paddle_sid].reshape(3, 3).T @ ref_face_dir

    # hit_pos, v_i, t_hit = compute_hit_pos(env.obs_dict["ball_pos"], env.obs_dict["ball_vel"])
    # v_o = compute_out_vel(target_pos=np.array([-0.685, 0.04, 0.795]), hit_pos=hit_pos)
    # v_r = compute_racket_vel(v_i, v_o)
    # # print(f"v_o:{v_o}")
    # # print(f"v_i:{v_i}")
    # # print(f"v_r:{v_r}")

    # # mocap_id = env.sim.model.body('direction_arrow').mocapid[0]
    # # init_vector = np.array([0,0,1])
    # direction = v_r / np.linalg.norm(v_r)
    # # rotation_quat = vec_to_quat(init_vector, direction)

    # # env.sim.data.mocap_pos[mocap_id] = hit_pos
    # # env.sim.data.mocap_quat[mocap_id] = rotation_quat

    # pad_face_local = np.array([0,0,-1])
    # rotation_paddle = vec_to_quat(pad_face_local, direction)






    # # env.sim.data.mocap_pos[0] = hit_pos
    # env.mj_render()


    
    # policy = policies.ActorCriticPolicy(observation_space=env.observation_space,
    #                                 action_space=env.action_space,
    #                                 lr_schedule=lambda _: th.finfo(th.float32).max,
    #                                 net_arch = [dict(pi=[1024, 1024, 512, 512, 256, 256], vf=[1024, 1024, 512, 512, 256, 256])],
    #                                 activation_fn=th.nn.SiLU,
    #                                 )
    # model = policy.load("/home/zwt/Projects/myo/MC25-Mani-LLab/Dagger_train/onpolicy/dagger_policy_9850048.zip")


    # flight_time = []
    # import imageio.v2 as imageio
    # writer = imageio.get_writer("test.mp4", fps=30)
    # cnt = 0
    # while True:
    #     if env.obs_dict["time"] >= t_hit  and not collide:
    #         # print(env.sim.data.site_xmat[env.id_info.paddle_sid])
    #         env.sim.data.qpos[-11:-7] = rotation_paddle
    #         # mujoco.mj_forward(env.sim.model.ptr, env.sim.data.ptr)
    #         # print(f"env.sim.data.xmat[env.id_info.paddle_bid]: {env.sim.data.xmat[env.id_info.paddle_bid].reshape(3, 3)}")

    #         rot = R.from_quat([rotation_paddle[1], rotation_paddle[2], rotation_paddle[3], rotation_paddle[0]])
    #         rot_mat = rot.as_matrix()
            

    #         pad_loacl = np.array([-0.07, 0, 0])
    #         # pad_world 
    #         pad_world = rot_mat.reshape(3, 3) @ pad_loacl
    #         env.sim.data.qpos[-14:-11] = hit_pos - pad_world
    #         # env.sim.data.qpos[-14:-11] = hit_pos
    #         print(f"hit_pos: {hit_pos}")
    #         print(f"pad_world: {pad_world}")
            
    #         env.sim.data.qvel[-12:-9] = v_r

    #         collide = True      
    #         mujoco.mj_forward(env.sim.model.ptr, env.sim.data.ptr)
    #         env.mj_render()
    #         # obs, reward, truncated, done, info = env.step(np.zeros(env.action_space.shape[0]))
    #         time.sleep(1)
        
    #     else:
    #         # action, _ = model.predict(obs)
    #         obs, reward, truncated, done, info = env.step(np.zeros(env.action_space.shape[0]))

    #     obs = th.tensor(obs, dtype=th.float32, device=device)
    #     # obs, reward, truncated, done, info = env.step(env.action_space.sample())

    #     # writer.append_data(env.render(mode="rgb_array"))
    #     if done or truncated:
    #         obs, _ = env.reset()
    #         collide = False
    #         ref_face_dir = np.array([-1, 0, 0])
    #         face_paddle = env.sim.data.site_xmat[env.id_info.paddle_sid].reshape(3, 3).T @ ref_face_dir

    #         hit_pos, v_i, t_hit = compute_hit_pos(env.obs_dict["ball_pos"], env.obs_dict["ball_vel"])
    #         v_o = compute_out_vel(target_pos=np.array([-0.685, 0.04, 0.795]), hit_pos=hit_pos)
    #         v_r = compute_racket_vel(v_i, v_o)


    #         # print(v_r)
    #         # mocap_id = env.sim.model.body('direction_arrow').mocapid[0]
    #         # init_vector = np.array([0,0,1])
    #         direction = v_r / np.linalg.norm(v_r)
    #         # rotation_quat = vec_to_quat(init_vector, direction)

    #         env.sim.data.mocap_pos[0] = hit_pos
    #         # env.sim.data.mocap_quat[mocap_id] = rotation_quat


    #         pad_face_local = np.array([0,0,-1])
    #         rotation_paddle = vec_to_quat(pad_face_local, direction)


    #         # env.sim.data.mocap_pos[0] = hit_pos

    #         cnt += 1
    #         print("End", cnt)
    #         if cnt > 100:
    #             break
    #     env.mj_render()
    #     time.sleep(0.02)
    
    # writer.close()
