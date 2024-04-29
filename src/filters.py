import numpy as np

def kalaman_filter(mean, cov, controls, measurements, delta_t):
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
    
    # state transition matrix A (nxn): n is the number of state variables
    A = np.identity(len(mean))
    # get orientation of the robot
    theta = mean[2]
    # Control matrix B (nxm), where m is the number of control inputs
    B = np.array([[delta_t * np.cos(theta), 0],
                [delta_t * np.sin(theta), 0],
                [0, delta_t]]
                )
    # measurement noise (nxn), initialize with random values
    R = np.diag(np.random.rand(len(mean)))
    # measurement matrix C (kxn), where k is the number of measurements
    C = np.identity(len(mean))
    # measurement noise (kxk), initialize with random values
    Q = np.diag(np.random.rand(len(mean)))
    
    # prediction step
    # update the mean
    mean = np.matmul(A, mean) + np.matmul(B, controls)
    # update covariance matrix
    cov = np.matmul(np.matmul(A, cov), np.transpose(A)) + R
    # correction step
    # Kalman gain
    K = np.matmul(np.matmul(cov, np.transpose(C)), np.linalg.inv(np.matmul(np.matmul(C, cov), np.transpose(C)) + Q))
    # update the mean
    mean = mean + np.matmul(K, measurements - np.matmul(C, mean))
    # update covariance matrix
    cov = np.matmul(np.identity(len(mean)) - np.matmul(K, C), cov)

    return mean, cov

if __name__ == "__main__":
    mean = np.array([1,2,3])
    # define covariance matrix of size 3x3, with random diagonal values in the range [0,1]
    cov = np.diag(np.random.rand(3))
    controls = np.array([1,2])
    measurements = np.array([1,2,3])
    delta_t = 1.0

    mean, cov = kalaman_filter(mean, cov, controls, measurements, delta_t)
    print(mean, cov)
    