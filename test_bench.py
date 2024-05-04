import numpy as np
import unittest
from main import run_saved_simulation
import src.filters as filters
#import environment
from unittest.mock import patch
import src.agent as robot_agent
import pickle as pkl
import logging

# logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
logger = logging.getLogger("TEST_BENCH")

class Experiments(unittest.TestCase):
    def setUp(self):
        filters.MEASUREMENT_NOISE = lambda: np.zeros(3)
        filters.R = np.zeros((3, 3))
        filters.Q = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
        robot_agent.HANDLE_SENSORDATA_MEMORIZE = False
        with open("tracker.pkl", "rb") as f:
            self.track = pkl.load(f)

    def test_measurement_noise(self):
        filters.MEASUREMENT_NOISE = lambda: np.random.normal(0, 1, 3)
        run_saved_simulation(delta_t=1,
                             graphGUI=None,
                             track=self.track,
                             num_landmarks=20)

    def test_sensor_noise(self):
        robot_agent.HANDLE_SENSORDATA_MEMORIZE = True
        #filters.R = np.abs(np.diag(np.random.normal(0,3,3)))
        filters.R = np.array([[1.23,0,0],[0,2.23,0],[0,0,8.230]])
        logger.debug("R:",filters.R)
        run_saved_simulation(delta_t=1,
                             graphGUI=None,
                             track=self.track,
                             num_landmarks=20)


if __name__ == "__main__":
    suite = unittest.TestSuite([Experiments('test_sensor_noise')])
    suite.run(unittest.TestResult())
