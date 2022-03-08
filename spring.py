import numpy as np
import matplotlib.pyplot as plt


def assignParam(springParam):
    S = yamaguchiSpring(springParam['s'],
                        springParam['k1'], springParam['k2'], springParam["q0"])
    return S


class yamaguchiSpring():
    def __init__(self, s: int, k1: float, k2, q0: float):
        """
        Parameters
        ----------
        s: 1 or -1
            sign
        k1: float
            stiffness
        k2: float
            stiffness
        q0: float
            coordinate at which there is no torque
        """
        if s is not None:
            self.s = s
        else:
            self.s = 1
        if k1 is not None:
            self.k1 = k1
        else:
            self.k1 = 0
        if k2 is not None:
            self.k2 = k2
        else:
            self.k2 = 0
        if q0 is not None:
            self.q0 = q0
        else:
            self.q0 = 0

    def torque(self, q):
        """
        Parameters
        ----------
        q: joint coordinate
        """
        s = self.s
        k1 = self.k1
        k2 = self.k2
        q0 = self.q0

        return s * k1 * np.exp(-k2 * (q - q0))

    def sign(self):
        return self.s

    def stiffness1(self):
        return self.s

    def stiffness2(self):
        return self.s

    def q0(self):
        return self.q0


class linearSpring():
    def __init__(self, k: float, l0: float, a: float, qa: float, b: float, qb: float):
        """
        Parameters
        ----------
        k: float
        l0: float
        """
        if k is not None:
            self.k = k
        else:
            self.k = 0
        if l0 is not None:
            self.l0 = l0
        else:
            self.l0 = 0
        if a is not None:
            self.a = a
        else:
            self.a = 0
        if b is not None:
            self.b = b
        else:
            self.b = 0
        if qa is not None:
            self.qa = qa
        else:
            self.qa = 0
        if qb is not None:
            self.qb = qb
        else:
            self.qb = 0

    def qTriangle(self, q):
        return 2 * np.pi - (q + self.qa + self.qb)

    def length(self, q):
        # return np.sqrt(self.a ** 2 + self.b ** 2 + 2 * self.a * self.b * np.cos(q + self.qa + self.qb))
        return np.sqrt(self.a ** 2 + self.b ** 2 - 2 * self.a * self.b * np.cos(self.qTriangle(q)))

    def force(self, q: float):
        """
        Parameters
        ----------
        q: angle
        """
        k = self.k
        l0 = self.l0

        return - k * (self.length(q) - l0)

    def momentArm(self, q):
        den = self.length(q)
        num = - self.a * self.b * np.sin( q + self.qa + self.qb)
        return - num / den

    def torque(self, q):
        R = self.momentArm(q)
        F = self.force(q)
        return R * F
