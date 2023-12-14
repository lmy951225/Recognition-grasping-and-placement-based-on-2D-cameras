from typing import List
import math

class JointPosition():
    def __init__(self, joints:List[float]) -> None:
        self._j1 = joints[0]
        self._j2 = joints[1]
        self._j3 = joints[2]
        self._j4 = joints[3]
        self._j5 = joints[4]
        self._j6 = joints[5]

     
    @property
    def j1(self) -> float:
        return self._j1

    @property
    def j2(self) -> float:
        return self._j2
    
    @property
    def j3(self) -> float:
        return self._j3
    
    @property
    def j4(self) -> float:
        return self._j4
    
    @property
    def j5(self) -> float:
        return self._j5
    
    @property
    def j6(self) -> float:
        return self._j6
    
    def is_nearly_equal(self, other: "JointPosition", threshold: float=0.0002) -> bool:
        if abs(self._j1 - other.j1) < threshold and \
            abs(self._j2 - other.j2) < threshold and \
            abs(self._j3 - other.j3) < threshold and \
            abs(self._j4 - other.j4) < threshold and \
            abs(self._j5 - other.j5) < threshold and \
            abs(self._j6 - other.j6) < threshold:
            return True
        else:
            return False


class CartesianPosition():
    def __init__(self, xyzrxryrz: List[float]) -> None:
        self._x = xyzrxryrz[0]
        self._y = xyzrxryrz[1]
        self._z = xyzrxryrz[2]
        self._rx= xyzrxryrz[3]
        self._ry= xyzrxryrz[4]
        self._rz= xyzrxryrz[5]

    @property
    def x(self) -> float:
        return self._x
    
    @property
    def y(self) -> float:
        return self._y
    
    @property
    def z(self) -> float:
        return self._z
    
    @property
    def rx(self) -> float:
        return self._rx
    
    @property
    def ry(self) -> float:
        return self._ry
    
    @property
    def rz(self) -> float:
        return self._rz