import numpy as np
import pygame
from pygame import Surface

from src.agent.sensor import SensorManager


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

        # Constant Components
        self.radius = radius

        # Gui Components
        self.color = (255, 140, 0)
        self.line = None
        self.body = None
        self.sensor_manager = None
        self.guided_line = None

    def standard_move(self,
                      vl: float,
                      vr: float,
                      delta_t: float
                      ):
        """
        Execute standard move
        """
        # when vl=vr=v
        if vl == vr:
            if vl != 0:
                # update positions
                self.pos_x += vl * delta_t * np.cos(self.theta)
                self.pos_y += vl * delta_t * np.sin(self.theta)

        else:
            # get angular velocity
            w = (vr - vl) / (2 * self.radius)
            # get the ICC radius
            R = self.radius * (vr + vl) / (vr - vl)
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

        # Draw
        self.sensor_manager.draw(surface)

        # Guided Line near the walls
        if self.guided_line:
            pygame.draw.line(surface, (0, 200, 150), (self.pos_x, self.pos_y),
                             (self.guided_line[0], self.guided_line[1]), width=int(self.radius / 10))
            self.guided_line = None

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
