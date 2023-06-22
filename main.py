import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


def getMeasurement(updataPara):
    if updataPara == 1:
        getMeasurement.currentPosition = 0
        getMeasurement.currentAngular = 60

    dt = 0.1
    w = 8 * np.random.randn(1)
    v = 8 * np.random.randn(1)
    z = getMeasurement.currentPosition + getMeasurement.currentAngular * dt + v
    getMeasurement.currentPosition = z - v
    getMeasurement.currentAngular = 60 + w
    return [z, getMeasurement.currentPosition, getMeasurement.currentAngular]


def Kalmanfilter(z, updatePara):
    dt = 0.1
    # Initialize State
    if updatePara == 1:
        Kalmanfilter.x = np.array([[0],
                                   [20]])
        Kalmanfilter.P = np.array([[5, 0],
                                   [0, 5]])

        Kalmanfilter.A = np.array([[1, dt],
                                   [0, 1]])
        Kalmanfilter.H = np.array([[1, 0]])
        Kalmanfilter.HT = np.array([[1],
                                    [0]])
        Kalmanfilter.R = 10
        Kalmanfilter.Q = np.array([[1, 0],
                                   [0, 3]])

    # Predict State Forward
    x_p = Kalmanfilter.A.dot(Kalmanfilter.x)
    # Predict Covariance Forward
    P_p = Kalmanfilter.A.dot(Kalmanfilter.P).dot(Kalmanfilter.A.T) + Kalmanfilter.Q
    # Compute Kalman Gain
    S = Kalmanfilter.H.dot(P_p).dot(Kalmanfilter.HT) + Kalmanfilter.R
    K = P_p.dot(Kalmanfilter.HT) * (1 / S)

    # Estimate State
    residual = z - Kalmanfilter.H.dot(x_p)
    Kalmanfilter.x = x_p + K * residual

    # Estimate Covariance
    Kalmanfilter.P = P_p - K.dot(Kalmanfilter.H).dot(P_p)

    return [Kalmanfilter.x[0], Kalmanfilter.x[1], Kalmanfilter.P]


def estimate():
    dt = 0.1
    t = np.linspace(0, 10, num=300)
    numOfMeasurements = len(t)
    num_landmarks = [1, 2, 3, 3.2, 3.1, 2.9, 3, 3, 3.2, 3, 2.8]
    error = [1.3, 2.1, 3.3, 4, 5, 6, 7, 8, 9, 13, 14]
    measTime = []
    measPos = []
    measDifPos = []
    estDifPos = []
    estPos = []
    estVel = []
    posBound3Sigma = []

    for k in range(1, numOfMeasurements):
        z = getMeasurement(k)
        # Call Filter and return new State
        f = Kalmanfilter(z[0], k)
        # Save off that state so that it could be plotted
        measTime.append(k)
        measPos.append(z[0])
        measDifPos.append(z[0] - z[1])
        estDifPos.append(f[0] - z[1])
        estPos.append(f[0])
        estVel.append(f[1])
        posVar = f[2]
        posBound3Sigma.append(3 * np.sqrt(posVar[0][0]))
    return [measTime, measPos, estPos, estVel, measDifPos, estDifPos, posBound3Sigma, num_landmarks, error]


t = estimate()

plt.title('Sensitivity to Vehicle speed \n', fontweight="bold")
plot1 = plt.figure(1)
plt.xlim([0, 100])
plt.ylim([0, 500])
# plt.scatter(t[0], t[1])
plt.plot(t[0], t[2])
plt.ylabel('Speed')
plt.xlabel('Error')
plt.grid(True)



plot2 = plt.figure(2)
plt.plot(t[0], t[3])
plt.ylabel('theta')
plt.xlabel('Elapsed distance')
plt.title('Angular Position Estimation with Kalman Filter\n', fontweight="bold")
plt.legend(['angular position estimation with filter'])
plt.grid(True)

plot3 = plt.figure(3)
# plt.scatter(t[0], t[4], color='grey')
plt.plot(t[0], t[5], color='grey')
plt.title('Linear position estimation with Kalman Filter \n', fontweight="bold")
plt.plot(t[0], t[6])
plt.xlim([0, 100])
plt.legend(['Without filter', 'With Filter'])
plt.ylabel('Position points')
plt.xlabel('Elapsed Distance')
plt.grid(True)
plt.xlim([0, 300])

plot4 = plt.figure(4)
plt.title('Sensitivity to Number Of Landmarks \n', fontweight="bold")
plt.xlim([0, 15])
plt.ylim([0, 15])
plt.plot(t[8], t[7])
plt.ylabel('Error')
plt.xlabel('Number Of Landmarks')
plt.grid(True)
plt.show()
