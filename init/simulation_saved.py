from init.utils.utils import draw_all
from src.utils.engine import simulate
import pygame


def run_saved_simulation(
        delta_t: float,
        track: list,
        agent=None,
        win=None,
        environment_surface=None,
        env=None,
        font=None
):
    if track is None or len(track) == 0:
        raise ValueError("No track data found")

    # Define variables
    delta_t_max = delta_t
    delta_t_curr = delta_t_max

    freeze = False
    time_step = 0
    for vl, vr in track:
        pygame.time.delay(1)

        flag = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                flag = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    freeze = not freeze

        if flag:
            break

        while freeze:
            pygame.time.delay(25)
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        freeze = not freeze

        # core logic starts here
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

        draw_all(win, environment_surface, agent, vl, vr, delta_t, freeze, time_step, font, env)

        # update the time ste
        time_step += 1

    pygame.quit()