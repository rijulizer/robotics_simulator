import pygame
import math
import numpy as np
from utils import seg_intersect
from environment import Environment


class SensorLine:
    def __init__(self, pos_x: float, pos_y: float, radius: float, robot_theta: float, theta: float, sensorlength: float,
                 i: int):
        """SensorLine class handles the individual sensor line segment

        Args:
            pos_x (float): X coordinate of Agent
            pos_y (float): Y coordinate of Agent
            radius (float): Radius of the Agent
            robot_theta (float): Angle of rotation of the agent
            theta (float): Angle at which the sensor of located on the robot
            sensorlength (float): length of the sensor line segment
            i (int): unique index(identifier) of the sensor
        """
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.radius = radius
        self.index = i
        self.robot_theta = robot_theta
        self.theta = theta
        self.ftheta = theta + robot_theta
        self.sensor_length = sensorlength
        self.body = None

    def get_sensorpt(self, robot_pt: list, line_pt1: np.ndarray, line_pt2: np.ndarray):
        """Calculates the length of the sensor coordinates from the start to the point of intersection
           with a wall.

        Args:
            robot_pt (list): coordinates of the robot
            line_pt1 (np.ndarray): Coordinates of one endpoint of the wall
            line_pt2 (np.ndarray): Coordinates of other endpoint of the wall

        Returns:
            _type_: distance between the start of sensor and the intersection with wall
        """
        sensorline_pt1 = np.array(
            [robot_pt[0] + self.radius * np.cos(self.ftheta), robot_pt[1] + self.radius * np.sin(self.ftheta)])
        sensorline_pt2 = np.array([robot_pt[0] + (self.radius + self.sensor_length) * np.cos(self.ftheta),
                                   robot_pt[1] + (self.radius + self.sensor_length) * np.sin(self.ftheta)])
        intersection_pt = seg_intersect(sensorline_pt1, sensorline_pt2, line_pt1, line_pt2)
        dist = np.sqrt((sensorline_pt1[0] - intersection_pt[0]) ** 2 + (sensorline_pt1[1] - intersection_pt[1]) ** 2)
        dist = max(min(dist, self.sensor_length), 0)
        dist = np.nan_to_num(dist, False, nan=0.0)
        return dist

    def update(self, pos_x, pos_y, robot_theta):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.robot_theta = robot_theta
        self.ftheta = self.theta + self.robot_theta

    def draw(self, surface):
        self.body = pygame.draw.line(surface, (0, 0, 0), (
        self.pos_x + self.radius * np.cos(self.ftheta), self.pos_y + self.radius * np.sin(self.ftheta)), (
                                     self.pos_x + (self.radius + self.sensor_length) * np.cos(self.ftheta),
                                     self.pos_y + (self.radius + self.sensor_length) * np.sin(self.ftheta)),
                                     width=int(self.radius / 10))


class SensorManager:
    def __init__(self, pos_x: float, pos_y: float, radius: float, theta: float, no_sensor: int, sensorlength: int,
                 environment: Environment):
        """SensorManager holds the instances for all the Sensor Lines, and controls their text indicating the 
        distance between the sensors and the walls in case of collision.

        Args:
            pos_x (float): X coordinate of Agent
            pos_y (float): Y coordinate of Agent
            radius (float): Radius of the Agent
            theta (float): Angle of rotation of the agent
            no_sensor (int): Number of sensors on the Agent
            sensorlength (int): Length of each sensor on the agent
            environment (Environment): The environment object to check the collision of sensor points against the wall.
        """
        self.pygame_font = pygame.font.SysFont('Comic Sans MS', 12)
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.radius = radius
        self.theta = theta
        self.no_sensor = no_sensor
        self.sensor_length = sensorlength
        self.sensor_points_rad = [(i * 2 * math.pi / (self.no_sensor)) for i in range(self.no_sensor)]
        self.deltalist = [0 if val < 3.14 else 10 for val in self.sensor_points_rad]
        self.sensor_read = [str(self.sensor_length) for i in range(self.no_sensor)]
        self.text_surface = [self.pygame_font.render(self.sensor_read[i], False, (0, 0, 0)) for i in
                             range(self.no_sensor)]
        self.sensors = []
        self.environment = environment
        self.previous_sensor_collision = False

        for i in range(self.no_sensor):
            sl = SensorLine(self.pos_x, self.pos_y, self.radius, self.theta, self.sensor_points_rad[i], sensorlength, i)
            self.sensors.append(sl)

    def reset_sensor(self):
        self.sensor_read = [str(self.sensor_length) for i in range(self.no_sensor)]
        self.text_surface = [self.pygame_font.render(self.sensor_read[i], False, (0, 0, 0)) for i in
                             range(self.no_sensor)]

    def updatePosition(self, pos_x, pos_y, theta):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.theta = theta
        for i in range(self.no_sensor):
            self.sensors[i].update(pos_x, pos_y, theta)

    def updateSensorText(self, index, value):
        old_val = int(self.sensor_read[index])
        value = min(old_val, value)
        self.sensor_read[index] = str(value)
        self.text_surface[index] = self.pygame_font.render(self.sensor_read[index], False, (1, 1, 1))

    def updateSensorStatus(self):
        """Checks the collision of the sensors lines against the wall.
           and updates the textual information in this case.
        """
        robot_pt = [self.pos_x, self.pos_y]
        if self.previous_sensor_collision:
            self.previous_sensor_collision = False
            self.reset_sensor()
        for line in self.environment.line_list:
            for sensorline in self.sensors:
                if line.body.colliderect(sensorline.body):
                    line_pt1 = np.array([line.start_x, line.start_y])
                    line_pt2 = np.array([line.end_x, line.end_y])
                    dist = sensorline.get_sensorpt(robot_pt, line_pt1, line_pt2)
                    dist = math.floor(dist) - 1
                    self.updateSensorText(sensorline.index, dist)
                    self.previous_sensor_collision = True
        #             print(f"[Debug]-[Sim]-sensor- Collision occured at distance {dist}")
        # print(f"[INFO]-[Sim]-sensor- updateSensorStatus invoked Collision Status {self.previous_sensor_collision}")

    def draw(self, surface):
        for i in range(self.no_sensor):
            self.sensors[i].draw(surface)
        self.updateSensorStatus()
        for i in range(self.no_sensor):
            surface.blit(self.text_surface[i], ((self.pos_x + (
                        self.radius + self.sensor_length + self.deltalist[i]) * np.cos(
                self.theta + self.sensor_points_rad[i])),
                                                (self.pos_y + (self.radius + self.sensor_length + self.deltalist[
                                                    i]) * np.sin(self.theta + self.sensor_points_rad[i]))))
