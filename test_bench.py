import numpy as np
import unittest
from main import run_experiments
import src.filters as filters
from src.utils import add_control_noise

# import environment
from unittest.mock import patch
import src.agent as robot_agent
import pickle as pkl
import logging

# logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")
logger = logging.getLogger("TEST_BENCH")


class Experiments(unittest.TestCase):
    def setUp(self):
        filters.MEASUREMENT_NOISE = lambda: np.zeros(3)
        filters.CONTROL_NOISE = lambda controls: controls
        filters.R = np.abs(np.diag(np.random.normal(1, 1, 3)))
        filters.Q = np.abs(np.diag(np.random.normal(1, 1, 3)))
        robot_agent.HANDLE_SENSORDATA_MEMORIZE = True
        with open("tracker.pkl", "rb") as f:
            self.track = pkl.load(f)

    def test_measurement_noise_mlandmark(self):
        filters.MEASUREMENT_NOISE = lambda: np.random.normal(0, 1, 3)
        run_experiments(
            self.track,
            num_landmarks=20,
            file_name_win="Exp_w_measurement_noise_mlandmark",
            exp_name="Exp_w_measurement_noise_mlandmark",
        )

    def test_measurement_noise_llandmark(self):
        filters.MEASUREMENT_NOISE = lambda: np.random.normal(0, 1, 3)
        run_experiments(
            self.track,
            num_landmarks=12,
            file_name_win="Exp_w_measurement_noise_llandmark",
            exp_name="Exp_w_measurement_noise_llandmark",
        )

    def test_sensor_noise_mlandmark(self):
        robot_agent.HANDLE_SENSORDATA_MEMORIZE = True
        filters.R = np.array([[1.23, 0, 0], [0, 2.23, 0], [0, 0, 8.230]])
        run_experiments(
            self.track,
            num_landmarks=20,
            file_name_win="Exp_sensor_noise_mlandmark",
            exp_name="Exp_sensor_noise_mlandmark",
        )

    def test_sensor_noise_llandmark(self):
        robot_agent.HANDLE_SENSORDATA_MEMORIZE = True
        filters.R = np.array([[1.23, 0, 0], [0, 2.23, 0], [0, 0, 8.230]])
        run_experiments(
            self.track,
            num_landmarks=12,
            file_name_win="Exp_sensor_noise_llandmark",
            exp_name="Exp_sensor_noise_llandmark",
        )

    def test_control_noise_mlandmark(self):
        filters.CONTROL_NOISE = lambda controls: add_control_noise(
            controls, alpha=[0.01, 0.01, 0.01, 0.01]
        )
        run_experiments(
            self.track,
            num_landmarks=20,
            file_name_win="Exp_control_noise_mlandmark",
            exp_name="Exp_control_noise_mlandmark",
        )

    def test_control_noise_llandmark(self):
        filters.CONTROL_NOISE = lambda controls: add_control_noise(
            controls, alpha=[0.01, 0.01, 0.01, 0.01]
        )
        run_experiments(
            self.track,
            num_landmarks=12,
            file_name_win="Exp_control_noise_llandmark",
            exp_name="Exp_control_noise_llandmark",
        )

    def test_good_case(self):
        run_experiments(
            self.track,
            num_landmarks=20,
            file_name_win="Exp_Good_Case",
            exp_name="Exp_Good_Case",
        )

    def test_less_landmarks(self):
        run_experiments(
            self.track,
            num_landmarks=8,
            file_name_win="Exp_less_landmarks",
            exp_name="Exp_less_landmarks",
        )

    def test_zero_noise_mlandmark(self):
        filters.R = np.zeros((3, 3))
        filters.Q = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
        run_experiments(
            self.track,
            num_landmarks=20,
            file_name_win="Exp_zero_noise_mlandmark",
            exp_name="Exp_zero_noise_mlandmark",
        )

    def test_zero_noise_llandmark(self):
        filters.R = np.zeros((3, 3))
        filters.Q = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
        run_experiments(
            self.track,
            num_landmarks=12,
            file_name_win="Exp_zero_noise_llandmark",
            exp_name="Exp_zero_noise_llandmark",
        )

    def test_q_noise_mlandmark(self):
        filters.Q = np.array([[1.3, 0, 0], [0, 4.9, 0], [0, 0, 12.33]])
        filters.R = np.zeros((3, 3))
        run_experiments(
            self.track,
            num_landmarks=20,
            file_name_win="Exp_q_noise_mlandmark",
            exp_name="Exp_q_noise_mlandmark",
        )

    def test_q_noise_llandmark(self):
        filters.Q = np.array([[1.3, 0, 0], [0, 4.9, 0], [0, 0, 12.33]])
        filters.R = np.zeros((3, 3))
        run_experiments(
            self.track,
            num_landmarks=12,
            file_name_win="Exp_q_noise_llandmark",
            exp_name="Exp_q_noise_llandmark",
        )


if __name__ == "__main__":
    # suite = unittest.TestSuite([Experiments("test_good_case"),Experiments("test_measurement_noise"),Experiments("test_sensor_noise"),Experiments("test_control_noise")])
    suite = unittest.TestSuite(
        [
            Experiments("test_good_case"),
            Experiments("test_less_landmarks"),
            Experiments("test_zero_noise_mlandmark"),
            Experiments("test_zero_noise_llandmark"),
            Experiments("test_measurement_noise_mlandmark"),
            Experiments("test_measurement_noise_llandmark"),
            Experiments("test_sensor_noise_llandmark"),
            Experiments("test_sensor_noise_mlandmark"),
            Experiments("test_control_noise_mlandmark"),
            Experiments("test_control_noise_llandmark"),
            Experiments("test_q_noise_mlandmark"),
            Experiments("test_q_noise_llandmark"),
        ]
    )
    runner = unittest.TextTestRunner()
    runner.run(suite)
