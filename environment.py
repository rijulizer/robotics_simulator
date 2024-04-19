import pygame
class Environments(pygame.sprite.Sprite):
    def __init__(self,):
        super().__init__()
    
    def draw_env(win):
        # define environment
        border_x = 100
        border_y = 100
        border_len = 800
        border_height = 600
        border_width = 5
        # define obstacle line-1
        line_1_start_pos_x = border_x + int(border_len/4)
        line_1_end_pos_x = border_x + int(border_len/4)
        line_1_start_pos_y = border_y # starts from the top
        line_1_end_pos_y = int((border_y + border_height) - border_height/3)
        # define obstacle line-1
        line_2_start_pos_x = border_x + 2 * int(border_len/4)
        line_2_end_pos_x = border_x + 2 * int(border_len/4)
        line_2_start_pos_y = border_y + border_height
        line_2_end_pos_y = border_y + int(border_height/3)
        # define obstacle line-1
        line_3_start_pos_x = border_x + 3 * int(border_len/4)
        line_3_end_pos_x = border_x + 3 * int(border_len/4)
        line_3_start_pos_y = border_y # starts from the top
        line_3_end_pos_y = int((border_y + border_height) - border_height/3)
        # define the rectangular border
        pygame.draw.rect(win, (0,0,0), (border_x, border_y, border_len, border_height), width=border_width)
        # draw obscale lines
        pygame.draw.line(win, (0,0,0), (line_1_start_pos_x, line_1_start_pos_y), (line_1_end_pos_x, line_1_end_pos_y))
        pygame.draw.line(win, (0,0,0), (line_2_start_pos_x, line_2_start_pos_y), (line_2_end_pos_x, line_2_end_pos_y))
        pygame.draw.line(win, (0,0,0), (line_3_start_pos_x, line_3_start_pos_y), (line_3_end_pos_x, line_3_end_pos_y))
