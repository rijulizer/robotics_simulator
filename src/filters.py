import numpy as np

C = np.identity(3)
Q = np.diag(np.random.normal(1,1,3))
R = np.diag(np.random.normal(1,1,3))
Q = np.abs(Q)
R = np.abs(R)

MEASUREMENT_NOISE = lambda: np.zeros(3)

def kalman_filter(mean, cov, controls, measurements, delta_t):
    """
    Implements the Kalman filter algorithm for state estimation.

    Args:
        mean (numpy.ndarray): The mean of the state estimate.
        cov (numpy.ndarray): The covariance matrix of the state estimate.
        controls (numpy.ndarray): The control inputs applied to the system.
        measurements (numpy.ndarray): The measurements obtained from the system.
        delta_t (float): The time step between control inputs.

    Returns:
        numpy.ndarray: The updated mean of the state estimate.
        numpy.ndarray: The updated covariance matrix of the state estimate.
    """
    measurements += MEASUREMENT_NOISE()
    # state transition matrix A (nxn): n is the number of state variables
    A = np.identity(len(mean))
    # get orientation of the robot
    theta = mean[2]
    # Control matrix B (nxm), where m is the number of control inputs
    B = np.array([[delta_t * np.cos(theta), 0],
                  [delta_t * np.sin(theta), 0],
                  [0, delta_t]]
                 )

    # prediction step
    # update the mean
    pred_mean = np.matmul(A, mean) + np.matmul(B, controls)
    # update covariance matrix
    pred_cov = np.matmul(np.matmul(A, cov), np.transpose(A)) + R
    # correction step
    # Kalman gain
    K = np.matmul(np.matmul(pred_cov, np.transpose(C)),
                  np.linalg.inv(np.matmul(np.matmul(C, pred_cov), np.transpose(C)) + Q))
    K = np.matmul(np.matmul(pred_cov, np.transpose(C)), np.linalg.inv(np.matmul(np.matmul(C, pred_cov), np.transpose(C)) + Q))
    # update the mean
    mean = pred_mean + np.matmul(K, measurements - np.matmul(C, pred_mean))
    # update covariance matrix
    cov = np.matmul(np.identity(len(pred_mean)) - np.matmul(K, C), pred_cov)

    return mean, cov, pred_mean


if __name__ == "__main__":
    mean = np.array([1, 2, 3])
    # define covariance matrix of size 3x3, with random diagonal values in the range [0,1]
    cov = np.diag(np.random.rand(3))
    controls = np.array([1, 2])
    measurements = np.array([1, 2, 3])
    delta_t = 1.0

    mean, cov = kalman_filter(mean, cov, controls, measurements, delta_t)
    print(mean, cov)
