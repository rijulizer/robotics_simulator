import pygame
import numpy as np
from src.GUI.dust import Dust


class Line:
    def __init__(self, win, startX, startY, endX, endY):
        self.body = pygame.draw.line(win, (0, 0, 0), (startX, startY), (endX, endY), width=5)
        self.win = win
        self.start_x = startX
        self.start_y = startY
        self.end_x = endX
        self.end_y = endY

    def change_color(self, color):
        self.body = pygame.draw.line(self.win, color, (self.start_x, self.start_y),
                                     (self.end_x, self.end_y), width=5)


class Environment:
    def __init__(self, win):
        self.landmarks = None
        self.line_list = []
        self.init_environment(win)
        self.dust = Dust((80, 61), 1, win.get_size())

    def init_environment(self, win):
        border_x, border_y = 100, 100
        border_len, border_height = 800, 600
        border_width = 5  # Currently unused

        # Create border lines
        self.create_border(win, border_x, border_y, border_len, border_height)

        # Define obstacle lines
        obstacle_positions = [1, 2, 3]  # Divisions of border for obstacle lines
        for position in obstacle_positions:
            self.create_obstacle_line(win, position, border_x, border_y, border_len, border_height, 4)

        # List of environment joint points
        self.points = self.calculate_joint_points(border_x, border_y, border_len, border_height, obstacle_positions)

    def create_border(self, win, x, y, length, height):
        # Rectangular border creation
        coordinates = [
            (x, y, x, y + height),  # Left side
            (x, y, x + length, y),  # Top side
            (x + length, y, x + length, y + height),  # Right side
            (x + length, y + height, x, y + height)  # Bottom side
        ]
        for coord in coordinates:
            self.line_list.append(Line(win, *coord))

    def create_obstacle_line(self, win, division, border_x, border_y, border_len, border_height, space=6):
        start_x = end_x = border_x + division * (border_len // space+2)
        if division % 2 == 1:
            start_y, end_y = border_y, (border_y + border_height - border_height // space) - 50
        else:
            start_y, end_y = border_y + border_height, (border_y + border_height // space) + 50
        self.line_list.append(Line(win, start_x, start_y, end_x, end_y))

    def calculate_joint_points(self, x, y, length, height, positions):
        points = [
            [x, y], [x + length, y], [x, y + height], [x + length, y + height]
        ]
        for pos in positions:
            points.extend([
                [x + pos * (length // 8), y],
                [x + pos * (length // 8), y + height - height // 6]
            ])
        return points

    def put_landmarks(self, win, number_of_landmarks=20):
        # put landmarks on the environment
        # TODO: now all points are put as landmarks, we can put only some of them randomly and experiment
        # Pick random points from the environment points
        #np.random.seed(14)
        self.points = np.array(self.points)
        random_points = self.points[np.random.choice(self.points.shape[0], number_of_landmarks, replace=False), :]
        self.landmarks = []
        for i, point in enumerate(random_points):
            pygame.draw.circle(win, (255, 0, 0), (point[0], point[1]), 7)
            # add signatures to the landmarks
            self.landmarks.append({
                "signature": i,
                "x": point[0],
                "y": point[1]
            })
