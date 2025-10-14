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
    计算球的落点
    TODO: 假设落点高度为h=0.806m,经验统计
    return: landing position (x, y, h), t_land, land_vel
    """
    g = -9.81

    # 计算碰撞时间
    a = 0.5 * g
    b = cur_vel[2]
    c = cur_pos[2] - h
    t = solve_quadratic(a,b,c)

    # 计算落点位置
    x = cur_pos[0] + cur_vel[0] * t
    y = cur_pos[1] + cur_vel[1] * t

    # 计算落点速度
    land_vel = np.array([cur_vel[0], cur_vel[1], cur_vel[2] + g * t])

    return np.array([x, y, h]), t, land_vel


def compute_land_after_hit(cur_pos, cur_vel, h=0.806):
    """
    计算球的落点
    TODO: 假设落点高度为h=0.806m,经验统计
    return: landing position (x, y, h), t_land, land_vel
    """
    g = -9.81

    # 计算是否与球网发生碰撞
    t_net = (0 - cur_pos[0]) / cur_vel[0]
    h_net = 0.5*g*t_net**2 + cur_vel[2]*t_net + cur_pos[2]
    if h_net < 0.965:
        return None, None, None

    # 计算碰撞时间
    a = 0.5 * g
    b = cur_vel[2]
    c = cur_pos[2] - h
    t = solve_quadratic(a,b,c)

    # 计算落点位置
    x = cur_pos[0] + cur_vel[0] * t
    y = cur_pos[1] + cur_vel[1] * t

    # 计算落点速度
    land_vel = np.array([cur_vel[0], cur_vel[1], cur_vel[2] + g * t])

    return np.array([x, y, h]), t, land_vel

def solve_quadratic(a, b, c):
    """
    求解二次方程 ax^2 + bx + c = 0
    :param a: 二次项系数
    :param b: 一次项系数
    :param c: 常数项
    :return: 方程的解（可能为实数或复数）
    """
    if a == 0:
        # 非二次方程处理
        if b == 0:
            raise ValueError
        return -c / b
   
    # 计算判别式
    delta = b**2 - 4*a*c
   
    if delta > 0:
        root1 = (-b + math.sqrt(delta)) / (2 * a)
        root2 = (-b - math.sqrt(delta)) / (2 * a)
        # print(root1, root2)
        return max(root1, root2)
    elif delta == 0:
        # 一个实数根
        root = -b / (2 * a)
        # print(root)
        return root
    else:
        # 两个复数根
        root1 = (-b + cmath.sqrt(delta)) / (2 * a)
        root2 = (-b - cmath.sqrt(delta)) / (2 * a)
        # print(root1,root2)
        return max(root1, root2)
    
def compute_hit_pos(cur_pos, cur_vel, hit_plane_x=1.8):
    """
    计算球在接球平面的落点位置，经过测试该planner还是有一定的误差，后续再考虑使用数值驱动的方法。
    TODO: 超参数 hit_plane_x 需要调节
    """

    g = -9.81
    land_pos, t_land, land_vel = compute_land(cur_pos, cur_vel)

    # TODO：假设没有碰撞损失，且落地后竖直速度反向
    vel_bounce = -land_vel[2] 

    # 计算到达接球平面的时间
    t_subhit = (hit_plane_x - land_pos[0]) / land_vel[0]
    t_subhit = np.round(t_land+t_subhit, 2) - t_land

    x_hit = t_subhit * land_vel[0] + land_pos[0]
    y_hit = land_pos[1] + land_vel[1] * t_subhit
    z_hit = 0.5 * g * t_subhit**2 + vel_bounce * t_subhit + land_pos[2]

    v_hit = np.array([land_vel[0], land_vel[1], vel_bounce+g*t_subhit])

    return np.array([x_hit, y_hit, z_hit]), v_hit, t_land+t_subhit


def compute_out_vel(target_pos, hit_pos, t_flight=0.29):
    """
    计算击球点的出球速度
    TODO: 超参数 t_flight 需要调节, 0.29是经验数字
    """

    g = np.array([0,0,-9.81])
    v_out = (target_pos - hit_pos) / t_flight - 0.5 * g * t_flight

    # v_out2 = (target_pos - hit_pos) / t_flight + 0.5 * 9.81 * t_flight
    # print(v_out2)
    return v_out


def compute_racket_vel(v_i, v_o, C_r=1):
    """
    TODO: 假设法线方向上的碰撞稀疏为1,即没有速度损失
    """
    u = (v_o - v_i) / np.linalg.norm(v_o - v_i)

    v_racket = (v_o @ u + v_i @ u) / (1+C_r) * u 

    return v_racket

def vec_to_quat(from_vec, to_vec):
    """
    计算将 from_vec 旋转到 to_vec 的四元数。
    这段代码的思路是计算绝对旋转，即从被旋转物体的初始未旋转位置(局部和全局坐标系重合)开始计算，计算出一个绝对旋转四元数
    """
    from_vec = from_vec / np.linalg.norm(from_vec)
    to_vec = to_vec / np.linalg.norm(to_vec)

    axis = np.cross(from_vec, to_vec)
    angle = np.arccos(np.dot(from_vec, to_vec))

    # 处理向量平行或反向的特殊情况
    if np.linalg.norm(axis) < 1e-6:
        if angle < 0.1:  # 几乎平行
            return np.array([1.0, 0.0, 0.0, 0.0]) # 无旋转 (w,x,y,z)
        else:  # 反向
            # 绕任意垂直轴旋转180度，这里选择X轴
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

    # # 计算拍子位置和速度
    # hit_pos, v_i, t_hit = compute_hit_pos(env.obs_dict["ball_pos"], env.obs_dict["ball_vel"])
    # v_o = compute_out_vel(target_pos=np.array([-0.685, 0.04, 0.795]), hit_pos=hit_pos)
    # v_r = compute_racket_vel(v_i, v_o)
    # # print(f"v_o:{v_o}")
    # # print(f"v_i:{v_i}")
    # # print(f"v_r:{v_r}")

    # # 可视化拍子朝向
    # # mocap_id = env.sim.model.body('direction_arrow').mocapid[0]
    # # init_vector = np.array([0,0,1])
    # direction = v_r / np.linalg.norm(v_r)
    # # rotation_quat = vec_to_quat(init_vector, direction)

    # # env.sim.data.mocap_pos[mocap_id] = hit_pos
    # # env.sim.data.mocap_quat[mocap_id] = rotation_quat

    # # 计算球拍的朝向
    # pad_face_local = np.array([0,0,-1])
    # rotation_paddle = vec_to_quat(pad_face_local, direction)






    # # env.sim.data.mocap_pos[0] = hit_pos
    # env.mj_render()


    # # 加载训练好的模型
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
    #         # TODO： 设置拍子朝向
    #         # print(env.sim.data.site_xmat[env.id_info.paddle_sid])
    #         env.sim.data.qpos[-11:-7] = rotation_paddle
    #         # mujoco.mj_forward(env.sim.model.ptr, env.sim.data.ptr)
    #         # print(f"env.sim.data.xmat[env.id_info.paddle_bid]: {env.sim.data.xmat[env.id_info.paddle_bid].reshape(3, 3)}")

    #         rot = R.from_quat([rotation_paddle[1], rotation_paddle[2], rotation_paddle[3], rotation_paddle[0]])
    #         rot_mat = rot.as_matrix()
            

    #         # 转换为pad的位置
    #         pad_loacl = np.array([-0.07, 0, 0])
    #         # pad_world 
    #         pad_world = rot_mat.reshape(3, 3) @ pad_loacl
    #         env.sim.data.qpos[-14:-11] = hit_pos - pad_world
    #         # env.sim.data.qpos[-14:-11] = hit_pos
    #         print(f"hit_pos: {hit_pos}")
    #         print(f"pad_world: {pad_world}")
            
    #         # 设置拍子速度
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
    #         # 计算拍子位置和速度
    #         hit_pos, v_i, t_hit = compute_hit_pos(env.obs_dict["ball_pos"], env.obs_dict["ball_vel"])
    #         v_o = compute_out_vel(target_pos=np.array([-0.685, 0.04, 0.795]), hit_pos=hit_pos)
    #         v_r = compute_racket_vel(v_i, v_o)

    #         # # 可视化拍子朝向
    #         # print(v_r)
    #         # mocap_id = env.sim.model.body('direction_arrow').mocapid[0]
    #         # init_vector = np.array([0,0,1])
    #         direction = v_r / np.linalg.norm(v_r)
    #         # rotation_quat = vec_to_quat(init_vector, direction)

    #         env.sim.data.mocap_pos[0] = hit_pos
    #         # env.sim.data.mocap_quat[mocap_id] = rotation_quat

    #         # 计算球拍朝向
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
