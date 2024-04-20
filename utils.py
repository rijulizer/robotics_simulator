import numpy as np



def is_trace_collision(old_points: list, new_points: list, env_lines: list):
    """Checks if the traces lines collide with environemtn objects with bigger step sizes 

    Args:
        old_points (list): points on object from old timestamp
        new_points (list): points on object from new timestamp
        env_lines (list)

    Returns:
        bool:
    """
    assert len(old_points) == len(new_points)
    trace_lines =  np.array([(start, end) for start, end in zip(old_points, new_points)])
    for i in trace_lines:
        for j in env_lines:
            line_array = np.array([(j.start_x, j.start_y), (j.end_x, j.end_y)])
            num = is_intersect(i[0], i[1], line_array[0], line_array[1])
            if num:
                return True
    return False

def point_line_distance(px, py, ax, ay, bx, by):
    """ Calculate the distance from a point (px, py) to a line defined by two points (ax, ay) and (bx, by). """
    # Vector from point A to point B
    ab = [bx - ax, by - ay]
    # Vector from point A to point P
    ap = [px - ax, py - ay]
    # Calculate the magnitude of AB vector
    ab_mag = np.sqrt(ab[0]**2 + ab[1]**2)
    # Normalize the AB vector
    ab_unit = [ab[0] / ab_mag, ab[1] / ab_mag]
    # Project vector AP onto vector AB to find the closest point
    proj_length = np.dot(ap, ab_unit)
    # Ensure the projection length is within the segment
    proj_length = max(0, min(ab_mag, proj_length))
    # Find the closest point using the projection length
    closest = [ax + ab_unit[0] * proj_length, ay + ab_unit[1] * proj_length]
    # Calculate the distance from the closest point to the point P
    dist = np.sqrt((closest[0] - px)**2 + (closest[1] - py)**2)
    return dist

def circle_line_intersect(pos_x, pos_y, radius, start_x, start_y, end_x, end_y):
    """ Determine if a circle intersects with a line segment. """
    distance = point_line_distance(pos_x, pos_y, start_x, start_y, end_x, end_y)
    return distance <= radius + 5


def check_collision(agent, env):
    collision_angles = []
    for line in env.line_list:
        if circle_line_intersect(agent.pos_x, agent.pos_y, agent.radius, line.start_x, line.start_y, line.end_x, line.end_y):
            res = calculate_angle(line, agent)
            collision_angles.append(res)
    return collision_angles


def calculate_angle(line, agent):
    # Calculate direction vectors for each line
    vector1 = (agent.x_end - agent.pos_x, agent.y_end - agent.pos_y)
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

    # Calculate cross product to determine the relative direction
    cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]
    angle_direction = np.sign(cross_product)  # +1 for left, -1 for right relative to agent's direction

    return angle_rad * angle_direction  # Return the signed angle


# Line Intersection Points Algorithm borrowed from https://www.cs.mun.ca/~rod/2500/notes/numpy-arrays/numpy-arrays.html
def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
def seg_intersect(a1,a2, b1,b2) :
    """_summary_

    Args:
        a1 (np array of floats): Coordinates of one end of Line segment a
        a2 (np array of floats): Coordinates of other end of Line segment a
        b1 (np array of floats): Coordinates of one end of Line segment b
        b2 (np array of floats): Coordinates of other end of Line segment a

    Returns:
        np array of floats: Coordinates of point of intersection of Line segment a and b
    """
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom) * db + b1

def is_intersect(a1, a2, b1, b2):
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