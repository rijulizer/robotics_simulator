import numpy as np
import pygame
from pygame import Surface

from src.agent.sensor import SensorManager
from src.utils import circle_intersectoins, atan2
from src.filters import kalman_filter


class Agent:
    def __init__(self,
                 pos_x: float,
                 pos_y: float,
                 radius: int,
                 theta: float
                 ):
        # Changeable Components
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.theta = theta

        # initial belief with known correspondence
        self.bel_pos_x = pos_x
        self.bel_pos_y = pos_y
        self.bel_theta = theta

        # Constant Components
        self.radius = radius

        # Gui Components
        self.color = (255, 140, 0)
        self.line = None
        self.body = None
        self.sensor_manager = None
        self.guided_line = None

        self.bel_cov = np.diag(np.random.rand(3))

    def standard_move(self,
                      vl: float,
                      vr: float,
                      delta_t: float
                      ):
        """
        Execute standard move
        """
        # Default values
        v = 0
        w = 0
        if vl == vr:
            if vl != 0:
                # update positions
                self.pos_x += vl * delta_t * np.cos(self.theta)
                self.pos_y += vl * delta_t * np.sin(self.theta)
                v = vl

        else:
            # get angular velocity
            w = (vr - vl) / (2 * self.radius)
            # get the ICC radius
            R = self.radius * (vr + vl) / (vr - vl)
            v = R * w
            # ICC coordinates
            ICC_x = self.pos_x - R * np.sin(self.theta)
            ICC_y = self.pos_y + R * np.cos(self.theta)

            delta_theta = w * delta_t
            # Define rotation matrix
            mat_rot = np.array([[np.cos(delta_theta), - np.sin(delta_theta), 0],
                                [np.sin(delta_theta), np.cos(delta_theta), 0],
                                [0, 0, 1]
                                ])
            mat_init_icc = np.array([self.pos_x - ICC_x, self.pos_y - ICC_y, self.theta]).reshape(3, 1)
            mat_shift_origin = np.array([ICC_x, ICC_y, delta_theta]).reshape(3, 1)

            new_pos = (np.matmul(mat_rot, mat_init_icc) + mat_shift_origin).flatten()

            # unwrap the new positions
            self.pos_x = new_pos[0]
            self.pos_y = new_pos[1]
            self.theta = new_pos[2]

        # update belief
        return v, w

    def collision_move(self,
                       vl: float,
                       vr: float,
                       delta_t: float,
                       collision_angles: list
                       ):
        """
        Execute collision move

        Assumption: suspend any rotational motion, the agent only glides sticking to the wall
            i,e. Only x-y changes theta remains same
        """
        # Default values
        v = 0
        w = 0

        # if collision happens with multiple walls it does not move only rotation is allowed
        if len(collision_angles) > 1:
            # get angular velocity
            w = (vr - vl) / (2 * self.radius)
            self.theta += w * delta_t
        else:
            # for single collision
            collision_angle = collision_angles[0][0]  # TODO: change the logic to handle list of theta
            if vl == vr:
                v = vl
                # no angular velocity
                w = 0
            else:
                # get angular velocity
                w = (vr - vl) / (2 * self.radius)
                # get the ICC radius
                R = self.radius * (vr + vl) / (vr - vl)
                # get linear velocity
                v = R * w

            # get the component of v in the direction of glide
            v = v * np.cos(collision_angle)
            # compute angle of velocity from reference frame
            beta = self.theta + collision_angle  # TODO: validate this idea wrt. different combinations

            # modify positions
            self.pos_x += v * np.cos(beta) * delta_t
            self.pos_y += v * np.sin(beta) * delta_t
            self.theta += w * delta_t

            # Guided Line
            self.guided_line = (
                self.pos_x + v * np.cos(beta) * delta_t * 50, self.pos_y + v * np.sin(beta) * delta_t * 50)

        return v, w

    def get_agent_stats(self):
        """
        Get agent Position
        """
        return {
            "pos_x": self.pos_x,
            "pos_y": self.pos_y,
            "theta": self.theta,
            "radius": self.radius,
        }

    def set_agent_stats(self, stats: dict):
        """
        Set agent position
        """
        self.pos_x = stats['pos_x']
        self.pos_y = stats['pos_y']
        self.theta = stats['theta']

    def get_points_circle(self, num: int):
        """"
        Get the points around the agent
        """
        angles = [(2 * np.pi * n) / num for n in range(num)]
        return [(self.pos_x + self.radius * np.cos(angle), self.pos_y + self.radius * np.sin(angle)) for angle in
                angles]

    def set_sensor(self,
                   sensor_manager: SensorManager):
        """
        Set Sensor Manager
        """
        self.sensor_manager = sensor_manager

    def draw(self,
             surface: Surface
             ):
        """
        Draw Agent Components
        """
        # update the position of the agent to the sensor manager
        self.sensor_manager.update_agent_sensor_info(self.get_agent_stats())
        # draw object and assign to self.body
        self.body = pygame.draw.circle(surface, self.color, (self.pos_x, self.pos_y), self.radius)
        pygame.draw.circle(surface, (0, 0, 0), (self.pos_x, self.pos_y), self.radius, 3)

        # Draw the direction line
        # End points of direction line
        x_end = self.pos_x + self.radius * np.cos(self.theta)
        y_end = self.pos_y + self.radius * np.sin(self.theta)
        self.line = pygame.draw.line(surface, (0, 0, 0), (self.pos_x, self.pos_y), (x_end, y_end),
                                     width=int(self.radius / 10))

        # Draw sensor lines
        self.sensor_manager.draw(surface)

        # Guided Line near the walls
        # if self.guided_line:
        #     pygame.draw.line(surface, (0, 200, 150), (self.pos_x, self.pos_y),
        #                      (self.guided_line[0], self.guided_line[1]), width=int(self.radius / 10))
        #     self.guided_line = None
        if len(self.sensor_manager.detected_landmarks) > 0:
            for landmark in self.sensor_manager.detected_landmarks:
                # draw line from agent to the landmark
                pygame.draw.line(surface, (248, 228, 35), (self.pos_x, self.pos_y), (landmark["x"], landmark["y"]),
                                 width=3)

    def get_sensor_measurement(self):
        """
        Get the belief of the agent from detected landmarks
        """
        measurement = np.zeros(3)
        # if there are less than two landmarks detected, then simply assume the last belief
        if len(self.sensor_manager.detected_landmarks) >= 2:
            # get the last 2 landmarks
            detected_landmarks = self.sensor_manager.detected_landmarks[-2:]
            beacon_1 = detected_landmarks[0]
            beacon_2 = detected_landmarks[1]
            # get the two points of intersection of the two circles
            pos_points = circle_intersectoins([beacon_1["x"], beacon_1["y"], beacon_1["range"]],
                                              [beacon_2["x"], beacon_2["y"], beacon_2["range"]])
            delta_min = np.inf
            for x, y in pos_points:
                delta = abs((atan2(beacon_1["x"] - x, beacon_1["y"] - y) - beacon_1["bearing"]) -
                            (atan2(beacon_2["x"] - x, beacon_2["y"] - y) - beacon_2["bearing"]))
                if delta < delta_min:
                    delta_min = delta
                    measurement[0] = round(x,3)
                    measurement[1] = round(y,3)
                    measurement[2] = round(atan2(beacon_1["x"] - measurement[0], beacon_1["y"] - measurement[1]) -
                                   beacon_1["bearing"], 1)
        return measurement

    def apply_filter(self, v: float, w: float, delta_t: float):
        """
        Apply Kalman Filter
        """
        # Initialize mean as the initial belief
        mean = np.array([self.bel_pos_x, self.bel_pos_y, self.bel_theta])
        controls = np.array([v, w])

        # Set the latest measurements by getting the current belief
        measurements = self.get_sensor_measurement()

        mean, cov = kalman_filter(mean, self.bel_cov, controls, measurements, delta_t)

        # Update the belief
        self.bel_pos_x = mean[0]
        self.bel_pos_y = mean[1]
        self.bel_theta = mean[2]
        self.bel_cov = cov

# if __name__ == "__main__":
#     # define agent
#     pos_x = 200
#     pos_y = 200
#     radius = 30
#     theta = 0
#     color = (255,0,0)
#     agent = Agent(pos_x, pos_y , radius, theta, color)
#     agent.move(0,0, 5)
#     agent.move(5,0, 5)
