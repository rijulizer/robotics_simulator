import numpy as np
#from src.environment import Line
import sympy as sp
import pygame

def euclidean_distance(x1, y1, x2, y2):
    """
    Calculate the Euclidean distance between two points
    :return: Euclidean distance
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_collision_point_line(old_points: list,
                             new_points: list,
                             env_lines: list,
                             agent: dict
                             ):
    """
    Get the collision point of the agent with the environment, collision detetcion for big time steps with trace lines
    """
    assert len(old_points) == len(new_points)

    trace_lines = np.array([(start, end) for start, end in zip(old_points, new_points)])
    collision_lines = []
    for i in trace_lines:
        for j in env_lines:
            line_array = np.array([(j.start_x, j.start_y), (j.end_x, j.end_y)])
            points = get_position_line_intersection(i[0], i[1], line_array[0], line_array[1])
            # if the intersection point is not None, then append the line and the intersection point
            if points:
                collision_lines.append((points, j))

    min_distance = float('inf')
    closest_point = None
    # Find the closest collision point
    for point in collision_lines:
        distance = euclidean_distance(agent['pos_x'], agent['pos_y'], point[0][0], point[0][1])
        if distance < min_distance:
            min_distance = distance
            closest_point = point
    return closest_point


def closest_point_on_line(px, py, ax, ay, bx, by):
    """
    Calculate the closest point on a line to a given point
    """
    # Convert points to numpy vectors
    p = np.array([px, py])
    a = np.array([ax, ay])
    b = np.array([bx, by])

    # Vector from a to b
    ab = b - a
    # Vector from a to p
    ap = p - a

    # Project vector ap onto ab to find the closest point
    epsilon = 1e-10
    t = np.dot(ap, ab) / (np.dot(ab, ab) + epsilon)

    t = max(0, min(1, t))  # Clamp t to the range [0, 1]
    closest = a + t * ab
    return closest


def push_back_from_collision(pos_x, pos_y, radius, start_x, start_y, end_x, end_y):
    """
    Push back the agent from the collision
    """
    closest = closest_point_on_line(pos_x, pos_y, start_x, start_y, end_x, end_y)
    direction_vector = np.array([pos_x, pos_y]) - closest
    dist_to_closest = np.linalg.norm(direction_vector)

    # Calculate the required push distance to just touch the line
    push_distance = radius - dist_to_closest + 1

    # Normalize the direction vector
    if dist_to_closest != 0:
        direction_vector /= dist_to_closest

    # Push back the circle
    new_pos_x = pos_x + push_distance * direction_vector[0]
    new_pos_y = pos_y + push_distance * direction_vector[1]

    return new_pos_x, new_pos_y


def point_line_distance(px, py, ax, ay, bx, by):
    """ Calculate the distance from a point (px, py) to a line defined by two points (ax, ay) and (bx, by). """
    # Vector from point A to point B
    ab = [bx - ax, by - ay]
    # Vector from point A to point P
    ap = [px - ax, py - ay]
    # Calculate the magnitude of AB vector
    ab_mag = np.sqrt(ab[0] ** 2 + ab[1] ** 2)
    # Normalize the AB vector
    ab_unit = [ab[0] / ab_mag, ab[1] / ab_mag]
    # Project vector AP onto vector AB to find the closest point
    proj_length = np.dot(ap, ab_unit)
    # Ensure the projection length is within the segment
    proj_length = max(0, min(ab_mag, proj_length))
    # Find the closest point using the projection length
    closest = [ax + ab_unit[0] * proj_length, ay + ab_unit[1] * proj_length]
    # Calculate the distance from the closest point to the point P
    dist = np.sqrt((closest[0] - px) ** 2 + (closest[1] - py) ** 2)
    return dist


def circle_line_intersect(pos_x, pos_y, radius, start_x, start_y, end_x, end_y):
    """ Determine if a circle intersects with a line segment. """
    distance = point_line_distance(pos_x, pos_y, start_x, start_y, end_x, end_y)
    return distance <= radius


def get_wall_collision_angle(agent: dict,
                             object_list: list):
    """
    Check if the agent collides with any of the environment objects
    :param agent: Agent object
    :param object_list: List with objects
    :return: collision_angles: List of tuples containing the angle of collision and the line object
    """
    collision_angles = []
    for line in object_list:
        if circle_line_intersect(agent["pos_x"],
                                 agent["pos_y"],
                                 agent["radius"],
                                 line.start_x,
                                 line.start_y,
                                 line.end_x,
                                 line.end_y
                                 ):
            res = calculate_angle(line, agent)
            line.change_color((255, 87, 51))
            collision_angles.append((res, line))
        else:
            line.change_color((0, 0, 0))
    return collision_angles


def calculate_angle(line,
                    agent: dict):
    """
    Calculate the angle of collision between the agent and the wall
    """
    # Calculate direction vectors for each line
    x_end = agent["pos_x"] + agent["radius"] * np.cos(agent["theta"])
    y_end = agent["pos_y"] + agent["radius"] * np.sin(agent["theta"])
    vector1 = (x_end - agent["pos_x"], y_end - agent["pos_y"])
    vector2 = (line.end_x - line.start_x, line.end_y - line.start_y)

    # Calculate dot product of vectors
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

    # Calculate cross product to determine the relative direction
    cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]
    angle_direction = np.sign(cross_product)

    # Calculate magnitudes of each vector
    magnitude1 = np.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = np.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    # Calculate the angle in radians
    if magnitude1 * magnitude2 == 0:
        angle_rad = 0  # Prevent division by zero, though not applicable here as magnitudes are non-zero
    else:
        angle_rad = np.arccos(dot_product / (magnitude1 * magnitude2))

    return angle_rad * angle_direction  # Return the signed angle


# Line Intersection Points Algorithm borrowed from https://www.cs.mun.ca/~rod/2500/notes/numpy-arrays/numpy-arrays.html
def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
def seg_intersect(a1, a2, b1, b2):
    """_summary_

    Args:
        a1 (np array of floats): Coordinates of one end of Line segment a
        a2 (np array of floats): Coordinates of other end of Line segment a
        b1 (np array of floats): Coordinates of one end of Line segment b
        b2 (np array of floats): Coordinates of other end of Line segment a

    Returns:
        np array of floats: Coordinates of point of intersection of Line segment a and b
    """
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom) * db + b1


def get_position_line_intersection(a1, a2, b1, b2):
    x1, y1 = a1
    x2, y2 = a2
    x3, y3 = b1
    x4, y4 = b2

    # Coefficients for equations
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1
    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3

    # Solving equations by determinant
    determinant = A1 * B2 - A2 * B1
    if determinant == 0:
        # Lines are parallel
        return None
    else:
        # The intersection point
        px = (B2 * C1 - B1 * C2) / determinant
        py = (A1 * C2 - A2 * C1) / determinant

        # Check if the intersection point is on both line segments
        if min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2) and \
                min(x3, x4) <= px <= max(x3, x4) and min(y3, y4) <= py <= max(y3, y4):
            return (px, py)
        return None

def atan2(x, y):
    """Extends the inverse tangent function to all four quadrants."""
    if x == 0:
        if y == 0:
            return 0
        else:
            return np.sign(y) * np.pi / 2
    elif x > 0:
        return np.arctan(float(y / x))
    else:
        return np.sign(y) * (np.pi - np.arctan(abs(float(y / x))))

def circle_intersectoins(circle_1_points, circle_2_points):
    """
    Calculate the intersection points of two circles.

    Parameters:
    circle_1_points (tuple): The center coordinates (x, y) and radius of the first circle.
    circle_2_points (tuple): The center coordinates (x, y) and radius of the second circle.
    
    Returns:
    list: A list of tuples representing the coordinates of the intersection points.
    """
    # circulate to get the point of intersection
    c1 = sp.Circle((circle_1_points[0], circle_1_points[1]), circle_1_points[2])
    c2 = sp.Circle((circle_2_points[0], circle_2_points[1]), circle_2_points[2])
    intersection_points = c1.intersection(c2)
    # get the coordinates of the intersection points
    intersection_points = [(point.x, point.y) for point in [point.evalf() for point in intersection_points]]
    return intersection_points

# check if a point lies on a line defined by two points using sympy
def point_on_line(point, line):
    """
    Check if a point lies on a line defined by two points.

    Parameters:
    point (tuple): The coordinates of the point.
    line (tuple): (Ex: ((0,0), (2,2)))The coordinates of the two points defining the line.
    
    Returns:
    bool: True if the point lies on the line, False otherwise.
    """
    # create the line using the two points
    line = sp.Line(line[0], line[1])
    # create the point using the coordinates
    point = sp.Point(point)
    # check if the point lies on the line
    return line.contains(point)

def circle_line_intersection(circle, line):
    """
    Check if a circle intersects a line defined by two points.

    Parameters:
    circle (tuple): The center coordinates (x, y) and radius of the circle.
    line (tuple): The coordinates of the two points defining the line.
    
    Returns:
    bool: True if the circle intersects the line, False otherwise.
    """
    # calculate the distance from the center of the circle to the line
    distance = point_line_distance(circle[0], circle[1], line[0][0], line[0][1], line[1][0], line[1][1])
    # return True if the distance is less than the radius of the circle
    return distance <= circle[2]

# agent belief ellipse
def draw_belief_ellipse(surface, bel_cov, bel_pos_x, bel_pos_y, scale):
    # draw the belief covariance ellipse
    if bel_cov is not None:
        cov = bel_cov[:2, :2] # asusmption always a diagonal matrix
        eigvals, eigvecs = np.linalg.eig(cov)
        # Find major and minor axes lengths
        major_axis = scale * np.sqrt(eigvals[0])
        minor_axis = scale * np.sqrt(eigvals[1])
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        ellipse_rect = pygame.Rect((0,0,major_axis,minor_axis))
        ellipse_surface = pygame.Surface(ellipse_rect.size,pygame.SRCALPHA)
        pygame.draw.ellipse(ellipse_surface, (240, 90, 90), (0, 0, major_axis, minor_axis), 5)
        ellipse_surface = pygame.transform.rotate(ellipse_surface, angle)
        surface.blit(ellipse_surface, ellipse_surface.get_rect(center = (bel_pos_x,bel_pos_y)))

def add_control_noise(controls,alpha = [0.0, 0.0, 0.0, 0.0]):
    """adds noise to the control inputs.
    Args:
        v (float): The forward velocity.
        w (float): The angular velocity.
    Returns:
        tuple: The control inputs with added noise.
    """
    #controls = [v, w]
    # zero cebtred gaussian noise where the standard deviation is proportional to the control inputs
    # alphas are global parameters
    vel_noise = np.random.normal(0, alpha[0] * np.abs(controls[0]) + alpha[1] * np.abs(controls[1]))
    angular_noise = np.random.normal(0, alpha[2] * np.abs(controls[0]) + alpha[3] * np.abs(controls[1]))
    controls[0] += vel_noise
    controls[1] += angular_noise
    return controls
