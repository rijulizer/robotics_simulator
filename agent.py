import numpy as np
import pygame

class Agent:
    def __init__(self, pos_x, pos_y , radius, theta, color):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.radius = radius
        self.theta = theta
        self.color = color
        self.line = None
        self.body = None
        self.x_end = self.pos_x + self.radius * np.cos(self.theta)
        self.y_end = self.pos_y + self.radius * np.sin(self.theta)
        self.guided_line = None
    
    def move(self, vl, vr, delta_t, collision_angles=[]):
        # when there is no collison do normal movement
        if not collision_angles:
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
        # case of collison 
        else:
            #Assumption: suspend any rotational motion, the agent only glides stickking to the wall
            # i,e. Only x-y changes theta remains same
            # if collision happens with multiple walls it doesnt not move only rotation is allowed
            if len(collision_angles) > 1:
                # get angular velocity
                w = (vr-vl)/(2 * self.radius)
                self.theta += w * delta_t
            else:
                # for single collision
                collision_angle = collision_angles[0] # TODO: chenge the logic to handle list of theta
                if (vl == vr):
                    v = vl
                    # no angular velocity
                    w = 0
                else:
                    # get angular velocity
                    w = (vr-vl)/(2 * self.radius)
                    # get the ICC radius
                    R = self.radius * (vr + vl) / (vr - vl)
                    # get linear velocity
                    v = R * w
                # get the component of v in the direction of glide
                v = v * np.cos(collision_angle)
                # compund angle of velocity from reference frame
                beta = self.theta + collision_angle # TODO: validate this idea wrt. different combinations

                # print(f"[debug]-[agent_move]- velocity: {v}, beta: {beta}")
                # modify positions
                self.pos_x += v * np.cos(beta) * delta_t
                self.pos_y += v * np.sin(beta) * delta_t
                self.theta += w * delta_t

                # Debug Guided Velocity
                self.guided_line = (self.pos_x + v * np.cos(beta) * delta_t * 50, self.pos_y + v * np.sin(beta) * delta_t * 50)




        
    def draw(self, surface, clear=False):
        # draw object and assign to self.body
        self.body = pygame.draw.circle(surface, self.color, (self.pos_x, self.pos_y), self.radius)
        # get the end point 
        self.x_end = self.pos_x + self.radius * np.cos(self.theta)
        self.y_end = self.pos_y + self.radius * np.sin(self.theta)
        # draw face line and assign it to line
        self.line = pygame.draw.line(surface, (0,0,0), (self.pos_x, self.pos_y), (self.x_end, self.y_end), width=int(self.radius/10))

        # debug
        if self.guided_line:
            self.guided_vel = pygame.draw.line(surface, (0, 200, 150), (self.pos_x, self.pos_y),
                                               (self.guided_line[0], self.guided_line[1]), width=int(self.radius / 10))
            if clear: self.guided_line = None

    def set_pos(self, pos: tuple):
        self.pos_x = pos[0]
        self.pos_y = pos[1]
        self.radius = pos[2]
        self.theta = pos[3]


# if __name__ == "__main__":
#     # define agent
#     pos_x = 200
#     pos_y = 200
#     radius = 30
#     theta = 0
#     color = (255,0,0)
#     agent = Agent(pos_x, pos_y , radius, theta, color)
#     agent.move(0,0, 5)
#     agent.move(5,0, 5)
#
# def move(self, vl, vr, delta_t):
#         # when vl=vr=v
#         if (vl == vr):
#             if vl != 0:
#                 # update positions
#                 self.pos_x += vl * delta_t * np.cos(self.theta)
#                 self.pos_y += vl * delta_t * np.sin(self.theta)
#
#         else:
#             # get angular velocity
#             w = (vr-vl)/(2 * self.radius)
#             # get the ICC radius
#             R = self.radius * (vr + vl) / (vr - vl)
#             # ICC cordinates
#             ICC_x = self.pos_x - R * np.sin(self.theta)
#             ICC_y = self.pos_y + R * np.cos(self.theta)
#
#             delta_theta = w * delta_t
#             # define rotation matrix
#             mat_rot = np.array([[np.cos(delta_theta), - np.sin(delta_theta), 0],
#                                 [np.sin(delta_theta), np.cos(delta_theta), 0],
#                                 [0, 0, 1]
#                                 ])
#             mat_init_icc = np.array([self.pos_x - ICC_x, self.pos_y - ICC_y, self.theta]).reshape(3,1)
#             mat_shift_origin = np.array([ICC_x, ICC_y, delta_theta]).reshape(3,1)
#
#             new_pos = (np.matmul(mat_rot, mat_init_icc) + mat_shift_origin).flatten()
#             # print("[DEBUG]-[Agent]-[move]- new_pos", new_pos, new_pos.shape)
#             # unwrap the new positions
#             self.pos_x = new_pos[0]
#             self.pos_y = new_pos[1]
#             self.theta = new_pos[2]