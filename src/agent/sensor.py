import numpy as np
from pygame import Surface
import pygame

from src.utils.utils import euclidean_distance, atan2, line_line_intersections


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

        self.agent_pos_x = agent_stats["pos_x"]
        self.agent_pos_y = agent_stats["pos_y"]
        self.radius = agent_stats["radius"]
        self.robot_theta = agent_stats["theta"]
        self.index = i
        self.theta = theta
        self.f_theta = theta + self.robot_theta
        self.sensor_length = sensor_length
        self.intersec_pts = None
        self.wall_dist = self.sensor_length + self.radius # intialize the distance to the sensor length
        self.body = None
        # for drawing the sensor line
        self.start_x = self.agent_pos_x + self.radius * np.cos(self.f_theta)
        self.start_y = self.agent_pos_y + self.radius * np.sin(self.f_theta)
        self.end_x = self.agent_pos_x + (self.radius + self.sensor_length) * np.cos(
            self.f_theta
        )
        self.end_y = self.agent_pos_y + (self.radius + self.sensor_length) * np.sin(
            self.f_theta
        )

        self.pygame_font = pygame.font.SysFont("Comic Sans MS", 12)
        self.sensor_text = str(self.sensor_length)
        self.text_surface = self.pygame_font.render(self.sensor_text, False, (0, 0, 0))

    def get_object_distance(self, object_list):
        """Calculates the point of intersection, and the distance between the sensor line and the wall
        Args:
            object_list (list): [Part of simulation] The object list to check the collision of sensor points against the wall.
        """
        self.wall_dist = None
        min_dist = self.sensor_length
        for line in object_list:
            env_line_pts = [(line.start_x, line.start_y), (line.end_x, line.end_y)]
            # this is the actual sensor line
            sensor_line_pts = [
                (
                    self.agent_pos_x + self.radius * np.cos(self.f_theta),
                    self.agent_pos_y + self.radius * np.sin(self.f_theta),
                ),
                (
                    self.agent_pos_x
                    + (self.radius + self.sensor_length) * np.cos(self.f_theta),
                    self.agent_pos_y
                    + (self.radius + self.sensor_length) * np.sin(self.f_theta),
                ),
            ]
            # get the intersection points, lines dont intersect it is None
            # intersec_pts = get_position_line_intersection(env_line_pts[0],env_line_pts[1], sensor_line_pts[0], sensor_line_pts[1])
            intersec_pts = line_line_intersections(env_line_pts, sensor_line_pts)
            # if the sensor line intersects with the wall
            if intersec_pts:
                dist = euclidean_distance(
                    sensor_line_pts[0][0], sensor_line_pts[0][1], intersec_pts[0], intersec_pts[1]
                )
                # if the sensor line intersects multiple walls get the nearest distance
                if dist < min_dist:
                    min_dist = dist
                    self.intersec_pts = intersec_pts
                    self.wall_dist = min_dist

    def update_sensor(self, agent_stats):
        """Update the sensor line information needed for drawing the sensor line
        Args:
            agent_stats (dict): agent stats
        """
        self.agent_pos_x = agent_stats["pos_x"]
        self.agent_pos_y = agent_stats["pos_y"]

        self.robot_theta = agent_stats["theta"]
        self.f_theta = self.theta + self.robot_theta
        self.start_x = self.agent_pos_x + self.radius * np.cos(self.f_theta)
        self.start_y = self.agent_pos_y + self.radius * np.sin(self.f_theta)
        # if the sensor line intersects with the wall
        if self.wall_dist:
            self.end_x = self.intersec_pts[0]
            self.end_y = self.intersec_pts[1]
            self.sensor_text = str(round(self.wall_dist,1))
        # if the sensor line does not intersect with the wall
        else:
            self.end_x = self.agent_pos_x + (self.radius + self.sensor_length) * np.cos(
                self.f_theta
            )
            self.end_y = self.agent_pos_y + (self.radius + self.sensor_length) * np.sin(
                self.f_theta
            )
            self.sensor_text = str(self.sensor_length)

        self.text_surface = self.pygame_font.render(self.sensor_text, False, (0, 0, 0))


class SensorManager:
    def __init__(self,
                 agent_stats: dict,
                 num_sensors: int,
                 sensor_length: int,
                 sim_object_list: list
                 ):
        """SensorManager holds the instances for all the Sensor Lines, and controls their text indicating the
        distance between the sensors and the walls in case of collision.

        Args:
            agent_stats (dict): agent stats
            num_sensor (int): Number of sensors on the Agent
            sensor_length (int): Length of each sensor on the agent
            sim_object_list (list): [Part of simulation] The object list to check the collision of sensor points against the wall.
        """
        self.agent_pos_x = agent_stats["pos_x"]
        self.agent_pos_y = agent_stats["pos_y"]
        self.agent_radius = agent_stats["radius"]
        self.agent_theta = agent_stats["theta"]
        self.num_sensors = num_sensors
        self.sensor_length = sensor_length
        self.sensor_thetas = [(i * 2 * np.pi / self.num_sensors) for i in range(self.num_sensors)]
        self.delta_list = [0 if val < 3.14 else 10 for val in self.sensor_thetas]
        
        self.sensors = []
        self.object_list = sim_object_list
        self.detected_landmarks = []
        
        # initialize the sensorlines
        for i in range(self.num_sensors):
            sl = SensorLine(agent_stats,
                            self.sensor_thetas[i],
                            sensor_length,
                            i)
            self.sensors.append(sl)

    def update(
            self, 
            agent_stats: dict):
        """Update the information of all the sensor lines
        Args:
            agent_stats (dict): agent stats
        """
        self.agent_pos_x = agent_stats["pos_x"]
        self.agent_pos_y = agent_stats["pos_y"]
        self.agent_radius = agent_stats["radius"]
        self.agent_theta = agent_stats["theta"]
        # udpate each sensor line related info
        for s in self.sensors:
            s.get_object_distance(self.object_list)
            # print(f"Sendor: ", s.index)
            s.update_sensor(agent_stats)
    
    def draw(self,
             surface: Surface,
             agent_stats: dict,
             ):
        """Draw the sensor lines on the screen
        Args:
            surface (Surface): The surface to draw the sensor lines on
            agent_stats (dict): agent stats
        """
        self.update(agent_stats)
        # self.update_sensor_status()
        for s in self.sensors:
            # draw the sesnor line
            s.body = pygame.draw.line(
                surface, 
                (0, 0, 0), 
                (s.start_x, s.start_y), 
                (s.end_x, s.end_y), 
                width=int(s.radius / 10)
            )
            # Draw the text on the sensor line
            surface.blit(s.text_surface, (s.end_x, s.end_y))

    def scan_landmarks(self,
                       landmarks: list,
                       time_step: int
                       ):
        """Scan the landmarks in the environment and get the range and bearing information"""
        
        detected_landmarks = []
        # iterate through all sensors and lands marks and check if the landmark point lies on the sensor line
        for landmark in landmarks:
            range_i = euclidean_distance(self.agent_pos_x, self.agent_pos_y, landmark["x"], landmark["y"])
            if range_i <= self.sensor_length:
                bear_i = atan2(landmark["x"] - self.agent_pos_x, landmark["y"] - self.agent_pos_y) - self.agent_theta
                time = next(
                    (i["time_step"] for i in self.detected_landmarks if i["signature"] == landmark["signature"]),
                    time_step)

                detected_landmarks.append({
                    "x": landmark["x"],
                    "y": landmark["y"],
                    "signature": landmark["signature"],
                    "range": range_i,
                    "bearing": bear_i,
                    "time_step": time
                })
        self.detected_landmarks = sorted(detected_landmarks, key=lambda x: x["time_step"])

        