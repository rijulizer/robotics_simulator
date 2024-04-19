import numpy as np


def check_collision(agent, env):
    collision_angles = []
    for line in env.line_list:
        if line.body.colliderect(agent.body):
            res = calculate_angle(line, agent)
            collision_angles.append(res)
    return collision_angles


def calculate_angle(line, agent):
    # Calculate direction vectors for each line
    vector1 = (agent.x_end - agent.pos_x, agent.y_end - agent.pos_y)
    vector2 = (line.end_x - line.start_x, line.end_y - line.start_y)

    # Calculate dot product of vectors
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

    # Calculate magnitudes of each vector
    magnitude1 = np.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = np.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    # Calculate the angle in radians
    if magnitude1 * magnitude2 == 0:
        angle_rad = 0  # Prevent division by zero, though not applicable here as magnitudes are non-zero
    else:
        angle_rad = np.arccos(dot_product / (magnitude1 * magnitude2))

    # return np.degrees(angle_rad)
    return angle_rad