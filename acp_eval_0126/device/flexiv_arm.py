'''
Flexiv robot interface, built upon Flexiv RDK: https://github.com/FlexivArmics/flexiv_rdk/

Author: Hongjie Fang.
'''

import time
import flexivrdk
import numpy as np
from easydict import EasyDict as edict
import math

from easyrobot.arm.base import ArmBase
from easyrobot.utils.typing import Float32, List, Union, NDArray, Optional


class FlexivArm(ArmBase):
    '''
    Flexiv Arm Interface.
    '''

    _arm_repr = "quat"

    _base_transformation_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype = np.float32
    )
    
    _tcp_transformation_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype = np.float32
    )

    def __init__(
        self, 
        serial: str,
        logger_name: str = "Flexiv Arm",
        streaming_freq: int = 100, 
        shm_name: Optional[str] = None, 
        shm_freq: int = 30,
        **kwargs
    ) -> None:
        '''
        Initialization.
        
        Parameters:
        - serial: str, required, the serial of the robot;
        - logger_name: str, optional, default: "Flexiv Arm", the name of the logger;
        - streaming_freq: int, optional, default: 100, the arm states streaming frequency; 0 for disabling device states interface;
        - shm_name: str, optional, default: None, the shared memory name of the arm data, None means no shared memory object;
        - shm_freq: int, optional, default: 30, the shared memory updating frequency.
        '''
        self.mode = flexivrdk.Mode
        self.robot = flexivrdk.Robot(serial)
        if self.robot.fault():
            if not self.robot.ClearFault():
                raise RuntimeError("Cannot clear faults on the robot.")
        self.robot.Enable()
        while not self.robot.operational():
            time.sleep(0.5)
        self.current_mode = None

        super(FlexivArm, self).__init__(
            logger_name = logger_name,
            streaming_freq = streaming_freq,
            shm_name = shm_name,
            shm_freq = shm_freq,
            **kwargs
        )
    
    def _get_states(self, **kwargs):
        robot_states = self.robot.states()
        return edict({
            "tcp_pose": self._transform_pose(np.array(robot_states.tcp_pose, dtype = np.float32), to_control = False),
            "tcp_vel": self._transform_vel(np.array(robot_states.tcp_vel, dtype = np.float32), to_control = False),
            "joint_pos": np.array(robot_states.q, dtype = np.float32),
            "joint_vel": np.array(robot_states.dq, dtype = np.float32),
            "force_torque_tcp": self._transform_force_torque(np.array(robot_states.ext_wrench_in_tcp, dtype = np.float32), to_control = False)
        })
    
    def _get_tcp_pose(self, **kwargs):
        return self._transform_pose(np.array(self.robot.states().tcp_pose, dtype = np.float32), to_control = False)
    
    def _get_tcp_vel(self, **kwargs):
        return self._transform_vel(np.array(self.robot.states().tcp_vel, dtype = np.float32), to_control = False)
    
    def _get_joint_pos(self, **kwargs):
        return np.array(self.robot.states().q, dtype = np.float32)
    
    def _get_joint_vel(self, **kwargs):
        return np.array(self.robot.states().dq, dtype = np.float32)
    
    def _get_force_torque_tcp(self, **kwargs):
        return self._transform_force_torque(np.array(self.robot.states().ext_wrench_in_tcp, dtype = np.float32), to_control = False)

    def switch_mode(self, mode):
        mode = getattr(self.mode, mode)
        if mode != self.current_mode:
            self.robot.SwitchMode(mode)
            self.current_mode = mode

    def cali_sensor(self) -> None:
        """
        Calibrate sensors.
        """
        self.switch_mode("NRT_PRIMITIVE_EXECUTION")
        self.robot.ExecutePrimitive("ZeroFTSensor", dict())
        while not self.robot.primitive_states()["terminated"]:
            time.sleep(0.1)
    
    def set_joint_impedance(
        self,
        stiffness: Union[List[float], NDArray[Float32]],
        damping_ratio: Union[List[float], NDArray[Float32]] = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
    ) -> None:
        """
        Set joint impedance (stiffness and damping_ratio).
        """
        self.switch_mode("NRT_JOINT_IMPEDANCE")
        self.robot.SetJointImpedance(stiffness, damping_ratio)
    
    def set_cartesian_impedance(
        self,
        stiffness: Union[List[float], NDArray[Float32]],
        damping_ratio: Union[List[float], NDArray[Float32]] = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
    ) -> None:
        """
        Set cartesian impedance (stiffness and damping ratio).
        """
        self.switch_mode("NRT_CARTESIAN_MOTION_FORCE")
        self.robot.SetCartesianImpedance(stiffness, damping_ratio)

    def send_tcp_pose(
        self,
        pose: Union[List[float], NDArray[Float32]],
        max_vel = 0.5,
        max_acc = 2.0,
        max_angular_vel = 1.0,
        max_angular_acc = 5.0,
        blocking: bool = False
    ) -> None:
        pose_control = self._transform_pose(pose, to_control = True)
        self.switch_mode("NRT_CARTESIAN_MOTION_FORCE")
        self.robot.SendCartesianMotionForce(
            pose_control, 
            max_linear_vel = max_vel,
            max_angular_vel = max_angular_vel,
            max_linear_acc = max_acc,
            max_angular_acc = max_angular_acc
        )

        if blocking:
            self.wait_for_tcp_move(pose)
    
    def send_joint_pos(
        self,
        pos: Union[List[float], NDArray[Float32]],
        max_vel: Union[List[float], NDArray[Float32]] = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 
        max_acc: Union[List[float], NDArray[Float32]] = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        target_vel: Union[List[float], NDArray[Float32]] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        impedance: bool = True,
        blocking: bool = False,
        pose_unit: str = "rad",
        stiffness_ratio: float = 0.01
    ):  
        """
        Send joint position to the robot.

        Parameters:
        - pos: the target joint position of the robot;
        - max_vel: the maximum velocity of the robot;
        - max_acc: the maximum acceleration of the robot;
        - target_vel: the target velocity of the robot at the target joint position;
        - impedance: whether to use impedance control;
        - blocking: whether to wait for the robot to complete movements, only available if impedance is False.
        - pose_unit: the unit of the target position, "deg" or "rad", default: "rad";
        - stiffness_ratio: the ratio of the stiffness, default: 0.01.
        """
        if pose_unit == "deg":
            pos = [math.radians(x) for x in pos]
        elif pose_unit == "rad":
            pass
        else:
            raise ValueError(f"Invalid pose unit: {pose_unit}")

        if impedance:
            self.switch_mode("NRT_JOINT_IMPEDANCE")
            K_q_scaled = np.multiply(self.robot.info().K_q_nom, stiffness_ratio)
            self.robot.SetJointImpedance(K_q_scaled)
        else:
            self.switch_mode("NRT_JOINT_POSITION")
        
        self.robot.SendJointPosition(
            pos,
            target_vel,
            max_vel,
            max_acc
        )

        if blocking:
            if impedance:
                self.logger.warning("The blocking parameter is unavailable and thus ignored for impedance control.")
            else:
                self.wait_for_joint_move(pos)

    
    def stop(self) -> None:
        super(FlexivArm, self).stop()
        self.robot.Stop()

    def reset_joint_impedance(self):
        orig_mode = self.robot.mode()
        self.switch_mode("NRT_JOINT_IMPEDANCE")
        K_q_default = self.robot.info().K_q_nom
        self.robot.SetJointImpedance(K_q_default)
        self.switch_mode(orig_mode)

    def reset_cartesian_impedance(self):
        orig_mode = self.robot.mode()
        self.switch_mode("NRT_CARTESIAN_MOTION_FORCE")
        self.robot.SetCartesianImpedance([10000,10000,1000,1500,1500,1500],[0.7, 0.7, 0.7, 0.7, 0.7, 0.7])
        self.robot.SetMaxContactWrench([1000,1000,1000,500,500,500])
        self.switch_mode(orig_mode)

    def set_nullspace_posture(
        self, 
        nullspace_position: Union[List[float], NDArray[Float32]], 
        pose_unit: str = "rad"
    ) -> None:
        """
        Set nullspace posture.
        Parameters:
        - nullspace_position: the target nullspace position of the robot;
        - pose_unit: the unit of the nullspace position, "deg" or "rad", default: "rad";
        """
        if pose_unit == "deg":
            nullspace_position = [math.radians(x) for x in nullspace_position]
        elif pose_unit == "rad":
            pass
        else:
            raise ValueError(f"Invalid pose unit: {pose_unit}")
        
        self.switch_mode("NRT_CARTESIAN_MOTION_FORCE")
        q_min = self.robot.info().q_min
        q_max = self.robot.info().q_max
        joints_clipped = np.clip(nullspace_position, q_min, q_max).tolist()
        try:
            self.robot.SetNullSpacePosture(joints_clipped)
        except:
            print('NullSpace Posture deliver fails')
            pass

    def tool_set(self, tool_name: str) -> None:
        """
        Set tool.
        Parameters:
        - tool_name: the name of the tool;
        """
        tool = flexivrdk.Tool(self.robot)
        self.switch_mode("IDLE")
        tool.Switch(tool_name)