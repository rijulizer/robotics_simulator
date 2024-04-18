import numpy as np
import pygame

class Agent:
    def __init__(self, pos_x, pos_y , radius, theta, color):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.radius = radius
        self.theta = theta
        self.color = color
    
    def move(self, vl, vr, delta_t):
        # when vl=vr=v
        if (vl == vr):
            if vl != 0:
                # update positions
                self.pos_x += vl * delta_t * np.cos(self.theta)
                self.pos_y += vl * delta_t * np.sin(self.theta) 
        
        else:
            # get angular velocity
            w = (vr-vl)/(2 * self.radius)
            # get the ICC radius
            R = self.radius * (vr + vl) / (vr - vl)
            # ICC cordinates
            ICC_x = self.pos_x - R * np.sin(self.theta)
            ICC_y = self.pos_y + R * np.cos(self.theta)

            delta_theta = w * delta_t
            # define rotation matrix
            mat_rot = np.array([[np.cos(delta_theta), - np.sin(delta_theta), 0],
                                [np.sin(delta_theta), np.cos(delta_theta), 0],
                                [0, 0, 1]
                                ])
            mat_init_icc = np.array([self.pos_x - ICC_x, self.pos_y - ICC_y, self.theta]).reshape(3,1)
            mat_shift_origin = np.array([ICC_x, ICC_y, delta_theta]).reshape(3,1)

            new_pos = (np.matmul(mat_rot, mat_init_icc) + mat_shift_origin).flatten()
            # print("[DEBUG]-[Agent]-[move]- new_pos", new_pos, new_pos.shape)
            # unwrap the new positions
            self.pos_x = new_pos[0]
            self.pos_y = new_pos[1]
            self.theta = new_pos[2]
    
    def draw(self, surface):
        # draw object 
        pygame.draw.circle(surface, self.color, (self.pos_x, self.pos_y), self.radius)
        
        # get the end point 
        x_end = self.pos_x + self.radius * np.cos(self.theta)
        y_end = self.pos_y + self.radius * np.sin(self.theta)
        pygame.draw.line(surface, (0,0,0), (self.pos_x, self.pos_y), (x_end, y_end), width=int(self.radius/10))

if __name__ == "__main__":
    # define agent
    pos_x = 200
    pos_y = 200
    radius = 30
    theta = 0
    color = (255,0,0)
    agent = Agent(pos_x, pos_y , radius, theta, color)
    agent.move(0,0, 5)
    agent.move(5,0, 5)
    