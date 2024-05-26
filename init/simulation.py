from init.utils.utils import draw_all
from src.utils.engine import simulate
import pygame
import pickle as pkl


def run_simulation(
        delta_t: float,
        track: bool = False,
        agent=None,
        win=None,
        environment_surface=None,
        env=None,
        font=None
):

    # Define variables
    delta_t_max = delta_t
    delta_t_curr = delta_t_max
    vl = 0
    vl_max = 5  # * delta_t
    vl_min = - vl_max
    vr = 0
    vr_max = 5  # * delta_t
    vr_min = - vr_max

    sim_run = True
    freeze = False
    time_step = 0
    tracker = []

    while sim_run:
        pygame.time.delay(25)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sim_run = False
            # read movements
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w and not freeze:
                    vl += 1  # delta_t
                    vl = min(vl, vl_max)
                elif event.key == pygame.K_s and not freeze:
                    vl -= 1  # delta_t
                    vl = max(vl, vl_min)
                elif event.key == pygame.K_o and not freeze:
                    vr += 1  # delta_t
                    vr = min(vr, vr_max)
                elif event.key == pygame.K_l and not freeze:
                    vr -= 1  # delta_t
                    vr = max(vr, vr_min)
                elif event.key == pygame.K_SPACE:
                    freeze = not freeze

        # core logic starts here
        if not freeze:
            success = simulate(agent,
                               vr,
                               vl,
                               delta_t_curr,
                               env.line_list,
                               env.landmarks,
                               time_step,
                               True
                               )

            if not success:
                delta_t_curr -= 0.1
            else:
                delta_t_curr = delta_t_max

            # update the time ste
            time_step += 1

            if track:
                tracker.append((vl, vr))

        draw_all(win, environment_surface, agent, vl, vr, delta_t, freeze, time_step, font, env)

    # save the tracker
    if track:
        with open("data/records/tracker.pkl", "wb") as f:
            pkl.dump(tracker, f)

    pygame.quit()
