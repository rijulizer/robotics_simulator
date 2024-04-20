from src.agent.agent import Agent
from src.utils import *


def simulate(agent: Agent,
             object_list: list,
             vr: float,
             vl: float,
             delta_t_curr: float) -> bool:

    # Current agent position
    curr_pos = agent.get_agent_stats()

    # Get current circle points of the agent
    curr_points_circle = agent.get_points_circle(8)

    # Execute the standard move
    agent.standard_move(vr, vl, delta_t_curr)

    # Check on wall collisions
    collision_angles = get_wall_collision_angle(agent.get_agent_stats(),
                                                object_list)

    if collision_angles:
        # Reset the agent position
        agent.set_agent_stats(curr_pos)

        # Execute collision move (near the wall)
        agent.collision_move(vr, vl, delta_t_curr, collision_angles)

        # If still collision, then push back from the collision
        collision_angles = get_wall_collision_angle(agent.get_agent_stats(),
                                                    object_list)
        if collision_angles:
            line = collision_angles[0][1]
            new_x, new_y = push_back_from_collision(agent.pos_x, agent.pos_y, agent.radius,
                                                    line.start_x, line.start_y, line.end_x, line.end_y)
            agent.set_agent_stats({
                                    "pos_x": new_x,
                                    "pos_y": new_y,
                                    "theta": agent.theta
                                    })

    # Get next circle points of the agent
    next_points_circle = agent.get_points_circle(8)

    # Check for intermediate collisions with the environment (In case of big time step)
    collision_line = get_collision_point_line(curr_points_circle,
                                              next_points_circle,
                                              object_list,
                                              agent.get_agent_stats())

    if collision_line:
        # Push back the agent from the collision intersection
        new_x, new_y = push_back_from_collision(agent.pos_x, agent.pos_y, agent.radius,
                                                collision_line[0][0], collision_line[0][1], collision_line[0][0],
                                                collision_line[0][1])
        line = collision_line[1]  # Get the closest intersection line

        # Check if move from new to next position intersects with the line
        intersect_point = seg_intersect(np.array((agent.pos_x, agent.pos_y)),
                                        np.array((new_x, new_y)),
                                        np.array((line.start_x, line.start_y)),
                                        np.array((line.end_x, line.end_y))
                                        )
        # if intersect_point is not finite (meaning not intersection), then move to new position
        if not np.isfinite(intersect_point).all():
            agent.set_agent_stats({
                                    "pos_x": new_x,
                                    "pos_y": new_y,
                                    "theta": agent.theta
                                    })
        else:
            # Reset the agent position
            agent.set_agent_stats(curr_pos)
            return False

    return True
