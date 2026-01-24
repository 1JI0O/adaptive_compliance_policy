"""
Evaluation Agent.
"""

import os
import sys
import pathlib

# 1. 算出根目录
ROOT_DIR = str(pathlib.Path(__file__).parent.parent.absolute())
# 2. 算出 PyriteML 所在的目录
PYRITE_ML_DIR = os.path.join(ROOT_DIR, 'PyriteML')

# 将这两个都加入环境变量
sys.path.append(ROOT_DIR)
sys.path.append(PYRITE_ML_DIR)

if __name__ == "__main__":
    os.chdir(ROOT_DIR)

import time
import numpy as np
from eval.device.flexiv_arm import FlexivArm
from eval.device.camera import RealSenseRGBDCamera
from eval.utils.transformation import xyz_rot_transform
from eval.device.flexiv_gripper import FlexivGripper


class SingleArmAgent:
    """
    Evaluation single-arm agent with Flexiv arms, Dahuan gripper and an Intel RealSense RGB-D camera.
    
    Follow the implementation here to create your own real-world evaluation agent.
    """
    def __init__(
        self,
        robot_serial,
        gripper_port,
        camera_serial,
        max_contact_wrench = [30, 30, 30, 10, 10, 10],
        max_vel = 0.5,
        max_acc = 2.0,
        max_angular_vel = 1.0,
        max_angular_acc = 5.0,
        **kwargs
    ):
        # initialize
        self.robot = FlexivArm(robot_serial)
        self.gripper = FlexivGripper(gripper_port)
        self.camera_serial = camera_serial
        self.camera = RealSenseRGBDCamera(serial = camera_serial)
        self.intrinsics = self.camera.get_intrinsic()
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.max_angular_vel = max_angular_vel
        self.max_angular_acc = max_angular_acc
        
        self.last_gripper = 0

        # move to ready pose
        self.robot.send_tcp_pose(
            self.ready_pose,
            max_vel = self.max_vel,
            max_acc = self.max_acc,
            max_angular_vel = self.max_angular_vel,
            max_angular_acc = self.max_angular_acc
        )
        self.gripper.close_gripper()
        time.sleep(5)
        self.robot.cali_sensor()

        # set max contact wrench
        self.robot.robot.SetMaxContactWrench(max_contact_wrench)
    
    @property
    def ready_pose(self):
        return np.array([0.5, 0, 0.17, 0, 0, 1, 0], dtype=np.float32)

    # 获取相机观察到的
    def get_global_observation(self):
        _, colors, depths = self.camera.get_rgbd_images()
        return colors, depths
    
    # 获取机器人当前状态（本体感知），tcp 和夹爪宽度
    # 机械臂相关东西在 /device/arm.py
    def get_proprio(self, rotation_rep = "rotation_6d", rotation_rep_convention = None, with_joint = False):
        tcp_pose = self.robot.get_tcp_pose()
        tcp_pose = xyz_rot_transform(
            tcp_pose,
            from_rep = "quaternion",
            to_rep = rotation_rep,
            to_convention = rotation_rep_convention
        )
        if with_joint:
            joint_pos = self.robot.get_joint_pos()
        gripper_width = self.gripper.get_states()["width"]

        # 返回的是这个
        proprio = np.concatenate([tcp_pose, [gripper_width]], axis = 0)
    
        if with_joint:
            proprio_joint = np.concatenate([joint_pos, [gripper_width]], axis = 0)
            return proprio, proprio_joint
        else:
            return proprio

    def get_wrench()
        wrench = self.robot.get_force_torque_tcp()
        return wrench

    def action(self, action,stiffness_vector, rotation_rep = "rotation_6d", rotation_rep_convention = None):
        tcp_pose = xyz_rot_transform(
            action[: 9],
            from_rep = rotation_rep, 
            to_rep = "quaternion",
            from_convention = rotation_rep_convention
        )

        self.robot.set_cartesian_impedance(stiffness_vector)

        self.robot.send_tcp_pose(
            tcp_pose,
            max_vel = self.max_vel,
            max_acc = self.max_acc,
            max_angular_vel = self.max_angular_vel,
            max_angular_acc = self.max_angular_acc
        )
        time.sleep(0.1)
        
        # gripper_action = False
        # if abs(action[9] - self.last_gripper) >= 0.01:
        #     self.gripper.set_width(action[9])
        #     self.last_gripper = action[9]        
        #     gripper_action = True

        # if gripper_action:
        #     time.sleep(0.5)

    
    def stop(self):
        self.robot.stop()
        self.gripper.stop()
        self.camera.stop()

