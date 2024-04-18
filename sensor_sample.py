import pygame
import numpy as np
import math


def angle_of_vector(x, y):
    return pygame.math.Vector2(x, y).angle_to((1, 0))
    
def angle_of_line(x1, y1, x2, y2):
    return angle_of_vector(x2-x1, y2-y1)

class Line(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((200, 200))
        self.image.set_colorkey((0, 0, 0))
        self.sx = 0
        self.sy = 0
        self.fx = 200
        self.fy = 200
        pygame.draw.line(self.image, (1, 1, 1), (self.sx, self.sy), (self.fx, self.fy), 5)
        self.rect = self.image.get_rect(topleft = (100, 100))
        self.angle =  angle_of_line(self.sx,self.sy,self.fx,self.fy)

class Robot(pygame.sprite.Sprite):
    def __init__(self,x,y,r,centreX,centreY):
        super().__init__()
        self.image = pygame.Surface((x+r, x+r))
        self.image.set_colorkey((0, 0, 0))
        self.x = x
        self.y = y
        self.r = r
        self.centre_x = centreX
        self.centre_y = centreY

        pygame.draw.circle(self.image, (255, 0, 0), (self.x, self.y),self.r)
        self.rect = self.image.get_rect(center = (self.centre_x, self.centre_y))
        pygame.draw.line(self.image,(1, 1, 1),(self.x,self.y),((self.x + self.r),(self.y + 0)),1)
        self.angle = 0

class Sensor(pygame.sprite.Sprite):
    def __init__(self,robot:Robot,no_sensor:int,sensorlength:int):
        super().__init__()
        self.image = pygame.Surface((robot.x+robot.r+robot.y+sensorlength, robot.x+robot.r+robot.y+sensorlength))
        self.image.set_colorkey((0, 0, 0))
        self.pygame_font = pygame.font.SysFont('Comic Sans MS', 12)
        self.x = robot.x
        self.y = robot.y
        self.r = robot.r
        self.no_sensor = no_sensor
        sensor_length = sensorlength
        sensor_points_rad = [(i*2*math.pi/(no_sensor)) for i in range(no_sensor)]
        sensor_read = [str(sensor_length) for i in range(no_sensor)]
        text_surface = [self.pygame_font.render(sensor_read[i], False, (1, 1, 1)) for i in range(no_sensor)]
        
        
        self.rect = self.image.get_rect(center = (robot.centre_x, robot.centre_y))
        robot.rect.clamp_ip(self.rect)
        
        #print(sensor_points_rad)
        for i in range(no_sensor):
            pygame.draw.line(self.image,(1, 1, 1),((self.x + self.r*np.cos(sensor_points_rad[i])),
                                                 (self.y + self.r*np.sin(sensor_points_rad[i])))
                                             ,((self.x + (self.r+sensor_length)*np.cos(sensor_points_rad[i])),
                                                 (self.y + (self.r+sensor_length)*np.sin(sensor_points_rad[i]))),1)
            self.image.blit(text_surface[i],((self.x + (self.r+sensor_length+5)*np.cos(sensor_points_rad[i])),
                                                 (self.y + (self.r+sensor_length+5)*np.sin(sensor_points_rad[i]))))     


if __name__ == "__main__":
    pygame.init()
    
    screen = pygame.display.set_mode((600, 600))
    clock = pygame.time.Clock()
    
    running = True
    l = Line()
    m = Robot(x=50,y=50,r=20,centreX=25,centreY=25)
    s = Sensor(robot = m,no_sensor = 12, sensorlength = 20)
    group = pygame.sprite.Group([l, m, s])
    group_sensor = pygame.sprite.Group()
    
    while running:
        clock.tick(100)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    
        keys = pygame.key.get_pressed()
        oldx = m.rect.x
        oldy = m.rect.y
        m.rect.x += (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * 3
        m.rect.y += (keys[pygame.K_DOWN] - keys[pygame.K_UP]) * 3
        m.rect.clamp_ip(screen.get_rect())
        s.rect.clamp_ip(screen.get_rect())
        hit = pygame.sprite.collide_mask(m, l)
        m.angle = angle_of_line(oldx,oldy,m.rect.x,m.rect.y)
        angle1 = m.angle*180/math.pi
       
       
        screen.fill((255, 255, 255))
        if hit:
            m.rect.x = oldx
            m.rect.y = oldy
            movedir = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])
            m.rect.x += movedir * 5*np.sin(angle1)
            m.rect.y += movedir * 5*np.cos(angle1)
    
        s.rect.x = m.rect.x
        s.rect.y = m.rect.y
        group.draw(screen)
        pygame.display.flip()
       
    pygame.quit()