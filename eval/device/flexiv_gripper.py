'''
Flexiv Gripper Interface.

Author: Hongjie Fang.
'''

import time
import flexivrdk
import numpy as np
from easydict import EasyDict as edict

from easyrobot.arm.flexiv import FlexivArm
from easyrobot.gripper.base import GripperBase
from easyrobot.utils.typing import Optional, NDArray, Float32


class FlexivGripper(GripperBase):
    '''
    Flexiv Gripper API
    '''

    _min_width = 0.0
    _max_width = 0.1

    def __init__(
        self, 
        robot: FlexivArm,
        logger_name: str = "Flexiv Gripper",
        streaming_freq: int = 50, 
        shm_name: Optional[str] = None, 
        shm_freq: int = 20,
        gripper_name: str = "GripperFlexivModbus",
        is_init: bool = False,
        init_width: float = 0.099,
        **kwargs
    ) -> None:
        '''
        Initialization.

        Parameters:
        - robot: FlexivArm, the Flexiv robot arm instance;
        - logger_name: str, optional, default: "Flexiv Gripper", the name of the logger;
        - streaming_freq: int, optional, default: 50, the gripper states streaming frequency; 0: disable info interface;
        - shm_name: str, optional, default: None, the shared memory name of the gripper data, None means no shared memory object;
        - shm_freq: int, optional, default: 20, the shared memory updating frequency;
        - gripper_name: str, optional, default: "GripperFlexivModbus", the name of the gripper;
        - is_init: bool, optional, default: False, whether to initialize the gripper when starting;
        - init_width: float, optional, default: 0.099, the initial width.
        '''
        self.gripper = flexivrdk.Gripper(robot.robot)
        self.gripper.Enable(gripper_name)
        time.sleep(0.1)
        if is_init:
            self.gripper.Init()
            time.sleep(5)
        self.gripper.Move(init_width, 0.1, 3)
        self.last_width = init_width
        super(FlexivGripper, self).__init__(
            logger_name = logger_name,
            streaming_freq = streaming_freq,
            shm_name = shm_name,
            shm_freq = shm_freq,
            **kwargs
        )

    def _transform_width(self, width: float, to_control: bool = False) -> float:
        '''
        Transform gripper width to real-world width in meters.
        - to_control = True: gripper width in meters to gripper width signals.
        - to_control = False: gripper width signals to gripper width in meters.
        '''
        return np.clip(width, self._min_width, self._max_width)

    def set_width(self, width: float, speed: float = 0.1, force: float = 3) -> None:
        '''
        Set gripper width.
        '''
        width_signal = self._transform_width(width, to_control = True)
        self.gripper.Move(width_signal, speed, force)
        self.last_width = np.clip(width, 0, self._max_width)

    def open_gripper(self) -> None:
        '''
        Open the gripper.
        '''
        self.set_width(self._max_width)

    def close_gripper(self) -> None:
        '''
        Close the gripper.
        '''
        self.set_width(self._min_width)

    def _get_states(self) -> dict:
        return edict({
            "width": np.array([self._transform_width(self.gripper.states().width, to_control = False)], dtype = np.float32),
            "last_width": np.array([self.last_width], dtype = np.float32)
        })

    def _get_width(self, **kwargs) -> NDArray[Float32]:
        '''
        Get the gripper width.
        '''
        return np.array([self._transform_width(self.gripper.states().width, to_control = False)], dtype = np.float32)

    def _get_last_width(self, **kwargs) -> NDArray[Float32]:
        '''
        Get the gripper last width.
        '''
        return np.array([self.last_width], dtype = np.float32)