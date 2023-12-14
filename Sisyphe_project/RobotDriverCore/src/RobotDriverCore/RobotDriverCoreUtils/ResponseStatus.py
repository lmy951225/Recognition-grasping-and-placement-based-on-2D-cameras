from aenum import Enum, extend_enum

class ResponseStatus(Enum):

    NO_FILE_FOUND = (1, "找不到该文件")
    CSV2TRAJECTORY_FAIL = (2, "csv转轨迹失败")
    UPLOAD_TRAJECTORY_FAIL = (3, "上传轨迹失败")
    WAITING_TIMEOUT = (4, "等待时间超时")

    OK = (0, '成功')
    INCOMPATIBLE_VERSION = (-1, '版本不兼容')
    CONNECTION_TIMEOUT = (-3, '连接超时')
    INTERFACE_NOTIMPLEMENTED = (-4, "接口未实现")
    INDEX_OUT_OF_RANGE = (-5, "索引下标越界")
    UNSUPPORTED_FILETYPE = (-6, "不支持的文件类型")
    UNSUPPORTED_PARAMETER = (-7, "不支持的机器人参数")
    UNSUPPORTED_SIGNAL_TYPE = (-8, "不支持的IO信号类型")
    PROGRAM_NOT_FOUND = (-9, "找不到对应的程序")
    PROGRAM_POSE_NOT_FOUND = (-10, "找不到对应的程序点位")
    WRITE_PROGRAM_POSE_FAILED = (-11, "更新写入程序点位失败")
    GET_ALARM_CODE_FAILED = (-12, "访问报警服务获取报警码失败")
    WRONG_POSITION_INFO = (-13, "控制器返回错误的点位信息")
    UNSUPPORTED_TRATYPE = (-14, "不支持的运动类型")
    INVALID_DH_LIST = (-15, "错误的变换参数列表，请联系开发人员")
    INTERVAL_PORTS_MUST_NOTNONE = (-16, "时间间隔和输出端口列表和电平持续时间必须非空")
    INVALID_IP_ADRESS = (-17, "无效的IP地址")
    INVALID_DH_PARAMETERS = (-18, "无效的DH参数")
    INVALID_IO_LIST_PARAMETERS = (-18, "无效的IO列表参数")
    INVALID_PAYLOAD_INFO = (-19, "无效的负载信息")
    INVALID_FLYSHOT_CONFIG = (-20, "无效的飞拍配置参数")
    FAILED_TO_DOWNLOAD_SAME_NAME_FILE = (-21, "因重名不允许覆写")

    CONTROLLER_ERROR = (-254, "详情请联系开发人员")
    SERVER_ERR = (-255, '其他原因')

    @property
    def code(self) -> int:
        """get status code

        Returns:
            int: code integer
        """
        return self.value[0]
    
    @property
    def message(self) -> str:
        """get message

        Returns:
            str: message string
        """
        return self.value[1]
    
    def str2ResponseStatus(self, error_msg: str, error_code: int = 999) -> 'ResponseStatus':
        if error_msg not in ResponseStatus.__dict__:
                extend_enum(ResponseStatus, error_msg, (error_code, error_msg))                
        return ResponseStatus[error_msg]