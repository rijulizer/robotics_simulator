
import numpy as np
import unittest
from main import run_simulation
import src.filters as filters
#import environment
from unittest.mock import patch
import src.agent as robot_agent

class Experiments(unittest.TestCase):
    def setUp(self):
        filters.MEASUREMENT_NOISE = np.zeros(3)
        filters.R = np.zeros((3,3))
        filters.Q = np.array([[0.1,0,0],[0,0.1,0],[0,0,0.1]])
        robot_agent.HANDLE_SENSORDATA_MEMORIZE = False

    def test_measurement_noise(self):
         filters.MEASUREMENT_NOISE = np.array([0.99,0.63,3.22])
         episode = {"max_time_steps": 100}
         run_simulation(delta_t=1, graphGUI=None,episode = episode)

    def test_sensor_noise(self):
         episode = {"max_time_steps": 100}
         robot_agent.HANDLE_SENSORDATA_MEMORIZE
         filters.R = np.array([[0.99,0,0],[0,0.98,0],[0,0,1.99]])
         run_simulation(delta_t=1, graphGUI=None,episode = episode)

if __name__ == "__main__":
    suite = unittest.TestSuite([Experiments('test_measurement_noise'), Experiments('test_sensor_noise')])
    suite.run(unittest.TestResult())