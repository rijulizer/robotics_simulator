import math
import numpy as np
from pygame import Surface
import pygame
import logging

from src.utils import seg_intersect, euclidean_distance, atan2, point_on_line, circle_line_intersection


class SensorLine:
    def __init__(self,
                 agent_stats: dict,
                 theta: float,
                 sensor_length: float,
                 i: int
                 ):
        """SensorLine class handles the individual sensor line segment

        Args:
            agent_stats: Agent stats
            theta (float): Angle at which the sensor of located on the robot
            sensor_length (float): length of the sensor line segment
            i (int): unique index(identifier) of the sensor
        """
        self.pos_x = agent_stats["pos_x"]
        self.pos_y = agent_stats["pos_y"]
        self.radius = agent_stats["radius"]
        self.robot_theta = agent_stats["theta"]

        self.index = i
        self.theta = theta
        self.f_theta = theta + self.robot_theta
        self.sensor_length = sensor_length
        self.body = None

    def get_intersection_pt(self,
                            robot_pt: list,
                            line_pt1: np.ndarray,
                            line_pt2: np.ndarray
                            ):
        """Calculates the length of the sensor coordinates from the start to the point of intersection
           with a wall.

        Args:
            robot_pt (list): coordinates of the robot
            line_pt1 (np.ndarray): Coordinates of one endpoint of the wall
            line_pt2 (np.ndarray): Coordinates of other endpoint of the wall

        Returns:
            _type_: distance between the start of sensor and the intersection with wall
        """
        sensor_line_pt1 = np.array(
            [robot_pt[0] + self.radius * np.cos(self.f_theta), robot_pt[1] + self.radius * np.sin(self.f_theta)])
        sensor_line_pt2 = np.array([robot_pt[0] + (self.radius + self.sensor_length) * np.cos(self.f_theta),
                                    robot_pt[1] + (self.radius + self.sensor_length) * np.sin(self.f_theta)])
        intersection_pt = seg_intersect(sensor_line_pt1, sensor_line_pt2, line_pt1, line_pt2)
        dist = np.sqrt((sensor_line_pt1[0] - intersection_pt[0]) ** 2 + (sensor_line_pt1[1] - intersection_pt[1]) ** 2)
        dist = max(min(dist, self.sensor_length), 0)
        dist = np.nan_to_num(dist, False, nan=0.0)
        return dist

    def get_end_points(self):
        """Get the end points of the sensor line segment"""

        x1 = self.pos_x + self.radius * np.cos(self.f_theta)
        y1 = self.pos_y + self.radius * np.sin(self.f_theta)
        x2 = self.pos_x + (self.radius + self.sensor_length) * np.cos(self.f_theta)
        y2 = self.pos_y + (self.radius + self.sensor_length) * np.sin(self.f_theta)
        return x1, y1, x2, y2

    def update(self,
               agent_stats: dict
               ):
        self.pos_x = agent_stats["pos_x"]
        self.pos_y = agent_stats["pos_y"]
        self.robot_theta = agent_stats["theta"]
        self.f_theta = self.theta + self.robot_theta

    def draw(self,
             surface: Surface
             ):
        # get end points of sensor lines
        x1, y1, x2, y2 = self.get_end_points()
        self.body = pygame.draw.line(surface, (0, 0, 0), (x1, y1), (x2, y2), width=int(self.radius / 10))


class SensorManager:
    def __init__(self,
                 agent_stats: dict,
                 no_sensor: int,
                 sensor_length: int,
                 object_list: list
                 ):
        """SensorManager holds the instances for all the Sensor Lines, and controls their text indicating the
        distance between the sensors and the walls in case of collision.

        Args:
            agent_stats (dict): agent stats
            no_sensor (int): Number of sensors on the Agent
            sensor_length (int): Length of each sensor on the agent
            object_list (list): The object list to check the collision of sensor points against the wall.
        """
        self.pygame_font = pygame.font.SysFont('Comic Sans MS', 12)
        self.pos_x = agent_stats["pos_x"]
        self.pos_y = agent_stats["pos_y"]
        self.radius = agent_stats["radius"]
        self.theta = agent_stats["theta"]
        self.no_sensor = no_sensor
        self.sensor_length = sensor_length
        self.sensor_points_rad = [(i * 2 * math.pi / self.no_sensor) for i in range(self.no_sensor)]
        self.delta_list = [0 if val < 3.14 else 10 for val in self.sensor_points_rad]
        self.sensor_read = [str(self.sensor_length) for i in range(self.no_sensor)]
        self.text_surface = [self.pygame_font.render(self.sensor_read[i], False, (0, 0, 0)) for i in
                             range(self.no_sensor)]
        self.sensors = []
        self.object_list = object_list
        self.previous_sensor_collision = False

        self.detected_landmarks = []

        for i in range(self.no_sensor):
            sl = SensorLine(agent_stats,
                            self.sensor_points_rad[i],
                            sensor_length,
                            i)
            self.sensors.append(sl)

    def reset_sensor(self):
        self.sensor_read = [str(self.sensor_length) for i in range(self.no_sensor)]
        self.text_surface = [self.pygame_font.render(self.sensor_read[i], False, (0, 0, 0)) for i in
                             range(self.no_sensor)]

    def update_agent_sensor_info(self,
                                 agent_stats: dict
                                 ):
        self.pos_x = agent_stats["pos_x"]
        self.pos_y = agent_stats["pos_y"]
        self.theta = agent_stats["theta"]
        for i in range(self.no_sensor):
            self.sensors[i].update(agent_stats)

    def update_sensor_text(self,
                           index,
                           value
                           ):
        old_val = int(self.sensor_read[index])
        value = min(old_val, value)
        self.sensor_read[index] = str(value)
        self.text_surface[index] = self.pygame_font.render(self.sensor_read[index], False, (1, 1, 1))

    def update_sensor_status(self):
        """Checks the collision of the sensors lines against the wall.
           and updates the textual information in this case.
        """
        robot_pt = [self.pos_x, self.pos_y]
        if self.previous_sensor_collision:
            self.previous_sensor_collision = False
            self.reset_sensor()
        for line in self.object_list:
            for sensor_line in self.sensors:
                if line.body.colliderect(sensor_line.body):
                    line_pt1 = np.array([line.start_x, line.start_y])
                    line_pt2 = np.array([line.end_x, line.end_y])
                    dist = sensor_line.get_intersection_pt(robot_pt, line_pt1, line_pt2)
                    dist = math.floor(dist) - 1
                    self.update_sensor_text(sensor_line.index, dist)
                    self.previous_sensor_collision = True

    def draw(self,
             surface: Surface
             ):
        for i in range(self.no_sensor):
            self.sensors[i].draw(surface)
        self.update_sensor_status()
        for i in range(self.no_sensor):
            surface.blit(self.text_surface[i], ((self.pos_x + (
                    self.radius + self.sensor_length + self.delta_list[i]) * np.cos(
                self.theta + self.sensor_points_rad[i])),
                                                (self.pos_y + (self.radius + self.sensor_length + self.delta_list[
                                                    i]) * np.sin(self.theta + self.sensor_points_rad[i]))))

    def scan_landmarks(self,
                       landmarks: list,
                       time_step: int
                       ):
        """Scan the landmarks in the environment and get the range and bearing information
        
        """
        # iterate through all sensors and lands marks and check if the landmark point lies on the sensor line
        detected_landmarks = []
        for landmark in landmarks:
            range_i = euclidean_distance(self.pos_x, self.pos_y, landmark["x"], landmark["y"])

            # if the beacon is within the sensor range of the agent
            # TODO: two cases are posiible for senssing landomark
            # 1. only if sensor line intersects with beacon 2 then we detect beacon
            # 2, if the beacon is within the sensor range of the agent then we can detect the beacon
            # case -1
            # if range_i < self.sensor_length:
            #     for sensor in self.sensors:
            #         # check if the landmark circle intersects the sensor line, 
            #         # making the landmark a circle instead of a point circumvents some error induced by typecasting
            #         x1,y1,x2,y2 = sensor.get_end_points()
            #         # if point_on_line((int(landmark[0]), int(landmark[1])), ((int(x1),int(y1)),(int(x2),int(y2)))):
            #         if circle_line_intersection((int(landmark[0]), int(landmark[1]), 7), ((int(x1),int(y1)),(int(x2),int(y2)))):
            #             bear_i = atan2(landmark[0] - self.pos_x, landmark[1] - self.pos_y) - self.theta
            #             #TODO: Bearing can also be calculated from sensor angle of the particular sensor
            #             # append the range and bear information along with the cordinates and signature of the beacon
            #             detected_landmarks.append([landmark[0], landmark[1], landmark[2], range_i, bear_i])
            #             break
            # case -2
            if range_i <= self.sensor_length + self.radius:

                bear_i = atan2(landmark["x"] - self.pos_x, landmark["y"] - self.pos_y) - self.theta
                #TODO: Bearing can also be calculated from sensor angle of the particular sensor
                # append the range and bear information along with the cordinates and signature of the beacon

                time = next((i["time_step"] for i in self.detected_landmarks if i["signature"] == landmark["signature"]),
                            time_step)


                detected_landmarks.append({
                    "x": landmark["x"],
                    "y": landmark["y"],
                    "signature": landmark["signature"],
                    "range": range_i,
                    "bearing": bear_i,
                    "time_step": time
                })

        self.detected_landmarks = detected_landmarks
        if len(self.detected_landmarks) > 2:
            # remove the oldest landmark based on time
            self.detected_landmarks = sorted(self.detected_landmarks, key=lambda x: x["time_step"])
            self.detected_landmarks.pop(0)


        # print landmark with signature
        for i in self.detected_landmarks:
            print(f"Detected Landmark: {i['x'], i['y'], i['signature'], round(i['range'],2), round(i['bearing']* 180 / 3.14,2)}")
        return self.detected_landmarks
