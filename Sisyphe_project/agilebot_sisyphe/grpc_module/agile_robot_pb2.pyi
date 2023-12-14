from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AlgoRes(_message.Message):
    __slots__ = ["code", "log"]
    CODE_FIELD_NUMBER: _ClassVar[int]
    LOG_FIELD_NUMBER: _ClassVar[int]
    code: ErrorCode
    log: ResLog
    def __init__(self, code: _Optional[_Union[ErrorCode, _Mapping]] = ..., log: _Optional[_Union[ResLog, _Mapping]] = ...) -> None: ...

class CartesianPosition(_message.Message):
    __slots__ = ["rx", "ry", "rz", "x", "y", "z"]
    RX_FIELD_NUMBER: _ClassVar[int]
    RY_FIELD_NUMBER: _ClassVar[int]
    RZ_FIELD_NUMBER: _ClassVar[int]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    rx: float
    ry: float
    rz: float
    x: float
    y: float
    z: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ..., z: _Optional[float] = ..., rx: _Optional[float] = ..., ry: _Optional[float] = ..., rz: _Optional[float] = ...) -> None: ...

class ControllerIP(_message.Message):
    __slots__ = ["controller_ip"]
    CONTROLLER_IP_FIELD_NUMBER: _ClassVar[int]
    controller_ip: str
    def __init__(self, controller_ip: _Optional[str] = ...) -> None: ...

class DeleteFlyShotTrajRequest(_message.Message):
    __slots__ = ["name"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class ErrorCode(_message.Message):
    __slots__ = ["error_code"]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    error_code: int
    def __init__(self, error_code: _Optional[int] = ...) -> None: ...

class ExecuteFlyShotTrajRequest(_message.Message):
    __slots__ = ["kinematics_params", "method", "move_to_start", "name", "use_cache"]
    KINEMATICS_PARAMS_FIELD_NUMBER: _ClassVar[int]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    MOVE_TO_START_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    USE_CACHE_FIELD_NUMBER: _ClassVar[int]
    kinematics_params: kinematicsParams
    method: str
    move_to_start: bool
    name: str
    use_cache: bool
    def __init__(self, name: _Optional[str] = ..., method: _Optional[str] = ..., move_to_start: bool = ..., use_cache: bool = ..., kinematics_params: _Optional[_Union[kinematicsParams, _Mapping]] = ...) -> None: ...

class FakeInput(_message.Message):
    __slots__ = ["fake_info"]
    FAKE_INFO_FIELD_NUMBER: _ClassVar[int]
    fake_info: str
    def __init__(self, fake_info: _Optional[str] = ...) -> None: ...

class GetIOReply(_message.Message):
    __slots__ = ["code", "io_states", "log"]
    CODE_FIELD_NUMBER: _ClassVar[int]
    IO_STATES_FIELD_NUMBER: _ClassVar[int]
    LOG_FIELD_NUMBER: _ClassVar[int]
    code: ErrorCode
    io_states: _containers.RepeatedCompositeFieldContainer[IOState]
    log: ResLog
    def __init__(self, code: _Optional[_Union[ErrorCode, _Mapping]] = ..., log: _Optional[_Union[ResLog, _Mapping]] = ..., io_states: _Optional[_Iterable[_Union[IOState, _Mapping]]] = ...) -> None: ...

class GetIORequest(_message.Message):
    __slots__ = ["io_channels", "io_type"]
    IO_CHANNELS_FIELD_NUMBER: _ClassVar[int]
    IO_TYPE_FIELD_NUMBER: _ClassVar[int]
    io_channels: _containers.RepeatedScalarFieldContainer[int]
    io_type: int
    def __init__(self, io_channels: _Optional[_Iterable[int]] = ..., io_type: _Optional[int] = ...) -> None: ...

class GetJointPositionReply(_message.Message):
    __slots__ = ["code", "joints", "log"]
    CODE_FIELD_NUMBER: _ClassVar[int]
    JOINTS_FIELD_NUMBER: _ClassVar[int]
    LOG_FIELD_NUMBER: _ClassVar[int]
    code: ErrorCode
    joints: JointPosition
    log: ResLog
    def __init__(self, code: _Optional[_Union[ErrorCode, _Mapping]] = ..., log: _Optional[_Union[ResLog, _Mapping]] = ..., joints: _Optional[_Union[JointPosition, _Mapping]] = ...) -> None: ...

class GetNameReply(_message.Message):
    __slots__ = ["code", "log", "manipulator_type"]
    CODE_FIELD_NUMBER: _ClassVar[int]
    LOG_FIELD_NUMBER: _ClassVar[int]
    MANIPULATOR_TYPE_FIELD_NUMBER: _ClassVar[int]
    code: ErrorCode
    log: ResLog
    manipulator_type: ManipulatorType
    def __init__(self, code: _Optional[_Union[ErrorCode, _Mapping]] = ..., log: _Optional[_Union[ResLog, _Mapping]] = ..., manipulator_type: _Optional[_Union[ManipulatorType, _Mapping]] = ...) -> None: ...

class GetPoseReply(_message.Message):
    __slots__ = ["code", "log", "pose"]
    CODE_FIELD_NUMBER: _ClassVar[int]
    LOG_FIELD_NUMBER: _ClassVar[int]
    POSE_FIELD_NUMBER: _ClassVar[int]
    code: ErrorCode
    log: ResLog
    pose: CartesianPosition
    def __init__(self, code: _Optional[_Union[ErrorCode, _Mapping]] = ..., log: _Optional[_Union[ResLog, _Mapping]] = ..., pose: _Optional[_Union[CartesianPosition, _Mapping]] = ...) -> None: ...

class GetRobotProgramReply(_message.Message):
    __slots__ = ["code", "joints", "log", "poses"]
    CODE_FIELD_NUMBER: _ClassVar[int]
    JOINTS_FIELD_NUMBER: _ClassVar[int]
    LOG_FIELD_NUMBER: _ClassVar[int]
    POSES_FIELD_NUMBER: _ClassVar[int]
    code: ErrorCode
    joints: _containers.RepeatedCompositeFieldContainer[JointPosition]
    log: ResLog
    poses: _containers.RepeatedCompositeFieldContainer[CartesianPosition]
    def __init__(self, code: _Optional[_Union[ErrorCode, _Mapping]] = ..., log: _Optional[_Union[ResLog, _Mapping]] = ..., joints: _Optional[_Iterable[_Union[JointPosition, _Mapping]]] = ..., poses: _Optional[_Iterable[_Union[CartesianPosition, _Mapping]]] = ...) -> None: ...

class GetRobotProgramRequest(_message.Message):
    __slots__ = ["name"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class GetServerVersionReply(_message.Message):
    __slots__ = ["code", "controller_version", "log"]
    CODE_FIELD_NUMBER: _ClassVar[int]
    CONTROLLER_VERSION_FIELD_NUMBER: _ClassVar[int]
    LOG_FIELD_NUMBER: _ClassVar[int]
    code: ErrorCode
    controller_version: str
    log: ResLog
    def __init__(self, code: _Optional[_Union[ErrorCode, _Mapping]] = ..., log: _Optional[_Union[ResLog, _Mapping]] = ..., controller_version: _Optional[str] = ...) -> None: ...

class IOAddrs(_message.Message):
    __slots__ = ["addrs"]
    ADDRS_FIELD_NUMBER: _ClassVar[int]
    addrs: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, addrs: _Optional[_Iterable[int]] = ...) -> None: ...

class IOState(_message.Message):
    __slots__ = ["io", "io_type", "state"]
    IO_FIELD_NUMBER: _ClassVar[int]
    IO_TYPE_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    io: int
    io_type: int
    state: int
    def __init__(self, io: _Optional[int] = ..., state: _Optional[int] = ..., io_type: _Optional[int] = ...) -> None: ...

class IsFlyShotTrajValidReply(_message.Message):
    __slots__ = ["code", "log", "traj_planning_time"]
    CODE_FIELD_NUMBER: _ClassVar[int]
    LOG_FIELD_NUMBER: _ClassVar[int]
    TRAJ_PLANNING_TIME_FIELD_NUMBER: _ClassVar[int]
    code: ErrorCode
    log: ResLog
    traj_planning_time: str
    def __init__(self, code: _Optional[_Union[ErrorCode, _Mapping]] = ..., log: _Optional[_Union[ResLog, _Mapping]] = ..., traj_planning_time: _Optional[str] = ...) -> None: ...

class IsFlyShotTrajValidRequest(_message.Message):
    __slots__ = ["kinematics_params", "method", "name", "save_traj"]
    KINEMATICS_PARAMS_FIELD_NUMBER: _ClassVar[int]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    SAVE_TRAJ_FIELD_NUMBER: _ClassVar[int]
    kinematics_params: kinematicsParams
    method: str
    name: str
    save_traj: bool
    def __init__(self, name: _Optional[str] = ..., method: _Optional[str] = ..., save_traj: bool = ..., kinematics_params: _Optional[_Union[kinematicsParams, _Mapping]] = ...) -> None: ...

class JointPosition(_message.Message):
    __slots__ = ["J1", "J2", "J3", "J4", "J5", "J6"]
    J1: float
    J1_FIELD_NUMBER: _ClassVar[int]
    J2: float
    J2_FIELD_NUMBER: _ClassVar[int]
    J3: float
    J3_FIELD_NUMBER: _ClassVar[int]
    J4: float
    J4_FIELD_NUMBER: _ClassVar[int]
    J5: float
    J5_FIELD_NUMBER: _ClassVar[int]
    J6: float
    J6_FIELD_NUMBER: _ClassVar[int]
    def __init__(self, J1: _Optional[float] = ..., J2: _Optional[float] = ..., J3: _Optional[float] = ..., J4: _Optional[float] = ..., J5: _Optional[float] = ..., J6: _Optional[float] = ...) -> None: ...

class ManipulatorType(_message.Message):
    __slots__ = ["manipulator_type"]
    MANIPULATOR_TYPE_FIELD_NUMBER: _ClassVar[int]
    manipulator_type: str
    def __init__(self, manipulator_type: _Optional[str] = ...) -> None: ...

class OptimizeKinematicsParamsReply(_message.Message):
    __slots__ = ["code", "kinematics_params", "log", "traj_planning_time"]
    CODE_FIELD_NUMBER: _ClassVar[int]
    KINEMATICS_PARAMS_FIELD_NUMBER: _ClassVar[int]
    LOG_FIELD_NUMBER: _ClassVar[int]
    TRAJ_PLANNING_TIME_FIELD_NUMBER: _ClassVar[int]
    code: ErrorCode
    kinematics_params: kinematicsParams
    log: ResLog
    traj_planning_time: str
    def __init__(self, code: _Optional[_Union[ErrorCode, _Mapping]] = ..., log: _Optional[_Union[ResLog, _Mapping]] = ..., kinematics_params: _Optional[_Union[kinematicsParams, _Mapping]] = ..., traj_planning_time: _Optional[str] = ...) -> None: ...

class OptimizeKinematicsParamsRequest(_message.Message):
    __slots__ = ["method", "name"]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    method: str
    name: str
    def __init__(self, name: _Optional[str] = ..., method: _Optional[str] = ...) -> None: ...

class ResLog(_message.Message):
    __slots__ = ["res_log"]
    RES_LOG_FIELD_NUMBER: _ClassVar[int]
    res_log: str
    def __init__(self, res_log: _Optional[str] = ...) -> None: ...

class SetIOReply(_message.Message):
    __slots__ = ["code", "io_states", "log"]
    CODE_FIELD_NUMBER: _ClassVar[int]
    IO_STATES_FIELD_NUMBER: _ClassVar[int]
    LOG_FIELD_NUMBER: _ClassVar[int]
    code: ErrorCode
    io_states: _containers.RepeatedCompositeFieldContainer[IOState]
    log: ResLog
    def __init__(self, code: _Optional[_Union[ErrorCode, _Mapping]] = ..., log: _Optional[_Union[ResLog, _Mapping]] = ..., io_states: _Optional[_Iterable[_Union[IOState, _Mapping]]] = ...) -> None: ...

class SetIORequest(_message.Message):
    __slots__ = ["io_states"]
    IO_STATES_FIELD_NUMBER: _ClassVar[int]
    io_states: _containers.RepeatedCompositeFieldContainer[IOState]
    def __init__(self, io_states: _Optional[_Iterable[_Union[IOState, _Mapping]]] = ...) -> None: ...

class SetSpeedRatioRequest(_message.Message):
    __slots__ = ["speed_ratio"]
    SPEED_RATIO_FIELD_NUMBER: _ClassVar[int]
    speed_ratio: float
    def __init__(self, speed_ratio: _Optional[float] = ...) -> None: ...

class UploadFlyShotTrajRequest(_message.Message):
    __slots__ = ["joints", "name", "offset", "shot_flags", "traj_addrs"]
    JOINTS_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    SHOT_FLAGS_FIELD_NUMBER: _ClassVar[int]
    TRAJ_ADDRS_FIELD_NUMBER: _ClassVar[int]
    joints: _containers.RepeatedCompositeFieldContainer[JointPosition]
    name: str
    offset: _containers.RepeatedScalarFieldContainer[int]
    shot_flags: _containers.RepeatedScalarFieldContainer[bool]
    traj_addrs: _containers.RepeatedCompositeFieldContainer[IOAddrs]
    def __init__(self, name: _Optional[str] = ..., joints: _Optional[_Iterable[_Union[JointPosition, _Mapping]]] = ..., shot_flags: _Optional[_Iterable[bool]] = ..., offset: _Optional[_Iterable[int]] = ..., traj_addrs: _Optional[_Iterable[_Union[IOAddrs, _Mapping]]] = ...) -> None: ...

class kinematicsParams(_message.Message):
    __slots__ = ["joint_acc_coef", "joint_jerk_coef", "joint_vel_coef"]
    JOINT_ACC_COEF_FIELD_NUMBER: _ClassVar[int]
    JOINT_JERK_COEF_FIELD_NUMBER: _ClassVar[int]
    JOINT_VEL_COEF_FIELD_NUMBER: _ClassVar[int]
    joint_acc_coef: _containers.RepeatedScalarFieldContainer[float]
    joint_jerk_coef: _containers.RepeatedScalarFieldContainer[float]
    joint_vel_coef: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, joint_vel_coef: _Optional[_Iterable[float]] = ..., joint_acc_coef: _Optional[_Iterable[float]] = ..., joint_jerk_coef: _Optional[_Iterable[float]] = ...) -> None: ...
