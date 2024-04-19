import numpy as np
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

    return np.degrees(angle_rad)

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
    return (num / denom)*db + b1