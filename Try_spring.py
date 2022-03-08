import spring
import numpy as np
import matplotlib.pyplot as plt

# A = spring.yamaguchiSpring(-1, -2, 10, 0)
# q = np.linspace(-0.1, 1.57, 100)
# plt.plot(q, A.torque(q))
# plt.plot(0, A.torque(0), 'o')
# plt.show()

lS1 = spring.linearSpring(6380, 0.15, 0.0756482, 0.81017, 0.07844, 5*np.pi/6)
lS2 = spring.linearSpring(6380, 0.15, 0.12506, 1.16056, 0.07844, 5*np.pi/6)
print(lS1)

q = np.linspace(-0.1, 1.57/2, 100)
plt.plot(q * 180 / np.pi, lS.qTriangle(q)*180/np.pi, label='angle')
plt.plot(q * 180 / np.pi, lS.length(q), label='length')
plt.plot(q * 180 / np.pi, lS.momentArm(q), label='momentArm')
# plt.plot(q, lS.length(q))
# plt.plot(0, A.torque(0), 'o')
plt.show()

plt.figure(1)
plt.plot(q * 180 / np.pi, lS1.momentArm(q), label='momentArm1')
plt.plot(q * 180 / np.pi, lS2.momentArm(q), label='momentArm2')
plt.legend()
plt.show()

plt.figure(11)
plt.plot(q * 180 / np.pi, lS1.length(q), label='length1')
plt.plot(q * 180 / np.pi, lS2.length(q), label='length2')
plt.legend()
plt.show()

plt.figure(12)
plt.plot(q * 180 / np.pi, lS1.force(q), label='force1')
plt.plot(q * 180 / np.pi, lS2.force(q), label='force2')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(q * 180 / np.pi, lS1.torque(q), label='torque1')
plt.plot(q * 180 / np.pi, lS2.torque(q), label='torque2')
plt.legend()
plt.show()