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
        # set default paraneters (No noise)
        filters.MEASUREMENT_NOISE = lambda: np.zeros(3)
        filters.CONTROL_NOISE = lambda controls: controls
        filters.R = np.array([[0.1,0,0],[0,0.1,0],[0,0,0.1]])
        filters.Q = np.array([[0.1,0,0],[0,0.1,0],[0,0,0.1]])
        # filters.R = np.abs(np.diag(np.random.normal(0, 1, 3)))
        # filters.Q = np.abs(np.diag(np.random.normal(0, 1, 3)))
        robot_agent.HANDLE_SENSORDATA_MEMORIZE = True
        with open("tracker.pkl", "rb") as f:
            self.track = pkl.load(f)
    
    def test_default_noise_mlandmark(self):
        run_experiments(
            self.track,
            num_landmarks=20,
            file_name_win="Exp_default_noise_mlandmark",
            exp_name="Exp_default_noise_mlandmark",
        )

    def test_default_noise_llandmark(self):
        run_experiments(
            self.track,
            num_landmarks=6,
            file_name_win="Exp_default_noise_llandmark",
            exp_name="Exp_default_noise_llandmark",
        )

    def test_zero_noise_mlandmark(self):
        filters.R = np.zeros((3, 3))
        filters.Q = np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]])
        run_experiments(
            self.track,
            num_landmarks=20,
            file_name_win="Exp_zero_noise_mlandmark",
            exp_name="Exp_zero_noise_mlandmark",
        )

    def test_zero_noise_llandmark(self):
        filters.R = np.zeros((3, 3))
        filters.Q = np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]])
        run_experiments(
            self.track,
            num_landmarks=6,
            file_name_win="Exp_zero_noise_llandmark",
            exp_name="Exp_zero_noise_llandmark",
        )
    
    def test_control_noise_mlandmark(self):
        filters.CONTROL_NOISE = lambda controls: add_control_noise(
            controls, alpha=[0.6, 0.6, 0.6, 0.6]
        )
        run_experiments(
            self.track,
            num_landmarks=20,
            file_name_win="Exp_control_noise_mlandmark",
            exp_name="Exp_control_noise_mlandmark",
        )

    def test_control_noise_llandmark(self):
        filters.CONTROL_NOISE = lambda controls: add_control_noise(
            controls, alpha=[0.6, 0.6, 0.6, 0.6]
        )
        run_experiments(
            self.track,
            num_landmarks=6,
            file_name_win="Exp_control_noise_llandmark",
            exp_name="Exp_control_noise_llandmark",
        )
    def test_measurement_noise_mean_mlandmark(self):
        filters.MEASUREMENT_NOISE = lambda: np.array([20, 30, 0.2])
        run_experiments(
            self.track,
            num_landmarks=20,
            file_name_win="Exp_measurement_noise_mean_mlandmark",
            exp_name="Exp_measurement_noise_mean_mlandmark",
        )

    def test_measurement_noise_mean_llandmark(self):
        filters.MEASUREMENT_NOISE = lambda: np.array([20, 30, 0.2])
        run_experiments(
            self.track,
            num_landmarks=6,
            file_name_win="Exp_measurement_noise_mean_llandmark",
            exp_name="Exp_measurement_noise_mean_llandmark",
        )
    ############################################################################################################
    
    def test_obs_noise_mlandmark_belowMN(self):
        """ Sets the R value of kalamn filter """

        filters.MEASUREMENT_NOISE = lambda: np.array([5, 5, 0.01])
        filters.R = np.array([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 0.1]])

        run_experiments(
            self.track,
            num_landmarks=20,
            file_name_win="Exp_obs_noise_mlandmark_belowMN",
            exp_name="Exp_obs_noise_mlandmark_belowMN",
        )

    def test_obs_noise_mlandmark_aboveMN(self):
        """ Sets the R + MEASUREMENT_NOISE value of kalamn filter """
        filters.MEASUREMENT_NOISE = lambda: np.array([50, 50, 0.2])
        filters.R = np.array([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 0.1]])
        run_experiments(
            self.track,
            num_landmarks=20,
            file_name_win="Exp_obs_noise_mlandmark_aboveMN",
            exp_name="Exp_obs_noise_mlandmark_aboveMN",
        )

    def test_obs_noise_llandmark_belowMN(self):
        """ Sets the R value of kalamn filter """
        
        filters.MEASUREMENT_NOISE = lambda: np.array([5, 5, 0.01])
        filters.R = np.array([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 0.1]])
        run_experiments(
            self.track,
            num_landmarks=6,
            file_name_win="Exp_obs_noise_llandmark_belowMN",
            exp_name="Exp_obs_noise_llandmark_belowMN",
        )

    def test_obs_noise_llandmark_aboveMN(self):
        """ Sets the R + MEASUREMENT_NOISE value of kalamn filter """
        
        filters.MEASUREMENT_NOISE = lambda: np.array([50, 50, 0.2])
        filters.R = np.array([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 0.1]])
        run_experiments(
            self.track,
            num_landmarks=6,
            file_name_win="Exp_obs_noise_llandmark_aboveMN",
            exp_name="Exp_obs_noise_llandmark_aboveMN",
        )
    ############################################################################################################
    def test_process_noise_mlandmark_belowMN(self):
        """ Sets the Q + CONTROL_NOISEvalue of kalamn filter """

        filters.Q = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
        filters.CONTROL_NOISE = lambda controls: add_control_noise(
            controls, alpha=[0.1, 0.1, 0.1, 0.1]
        )

        run_experiments(
            self.track,
            num_landmarks=20,
            file_name_win="Exp_process_noise_mlandmark_belowMN",
            exp_name="Exp_process_noise_mlandmark_belowMN",
        )

    def test_process_noise_mlandmark_aboveMN(self):
        """ Sets the Q + MEASUREMENT_NOISE value of kalamn filter """
        filters.Q = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
        filters.CONTROL_NOISE = lambda controls: add_control_noise(
            controls, alpha=[0.6, 0.6, 0.6, 0.6]
        )
        run_experiments(
            self.track,
            num_landmarks=20,
            file_name_win="Exp_process_noise_mlandmark_aboveMN",
            exp_name="Exp_process_noise_mlandmark_aboveMN",
        )

    def test_process_noise_llandmark_belowMN(self):
        """ Sets the Q value of kalamn filter """
        
        filters.Q = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
        filters.CONTROL_NOISE = lambda controls: add_control_noise(
            controls, alpha=[0.1, 0.1, 0.1, 0.1]
        )
        run_experiments(
            self.track,
            num_landmarks=6,
            file_name_win="Exp_process_noise_llandmark_belowMN",
            exp_name="Exp_process_noise_llandmark_belowMN",
        )

    def test_process_noise_llandmark_aboveMN(self):
        """ Sets the Q + MEASUREMENT_NOISE value of kalamn filter """
        
        filters.Q = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
        filters.CONTROL_NOISE = lambda controls: add_control_noise(
            controls, alpha=[0.6, 0.6, 0.6, 0.6]
        )
        run_experiments(
            self.track,
            num_landmarks=6,
            file_name_win="Exp_process_noise_llandmark_aboveMN",
            exp_name="Exp_process_noise_llandmark_aboveMN",
        )

if __name__ == "__main__":
    # suite = unittest.TestSuite([Experiments("test_good_case"),Experiments("test_measurement_noise"),Experiments("test_sensor_noise"),Experiments("test_control_noise")])
    suite = unittest.TestSuite(
        [
            Experiments("test_default_noise_mlandmark"),
            Experiments("test_default_noise_llandmark"),

            Experiments("test_zero_noise_mlandmark"),
            Experiments("test_zero_noise_llandmark"),

            Experiments("test_measurement_noise_mean_mlandmark"),
            Experiments("test_measurement_noise_mean_llandmark"),

            Experiments("test_control_noise_mlandmark"),
            Experiments("test_control_noise_llandmark"),

            Experiments("test_obs_noise_mlandmark_belowMN"),
            Experiments("test_obs_noise_mlandmark_aboveMN"),

            Experiments("test_obs_noise_llandmark_belowMN"),
            Experiments("test_obs_noise_llandmark_aboveMN"),

            Experiments("test_process_noise_mlandmark_belowMN"),
            Experiments("test_process_noise_mlandmark_aboveMN"),

            Experiments("test_process_noise_llandmark_belowMN"),
            Experiments("test_process_noise_llandmark_aboveMN"),
        ]
    )
    runner = unittest.TextTestRunner()
    runner.run(suite)
