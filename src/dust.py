import math
import numpy as np
from pygame import Surface
import pygame
import logging
from src.utils import euclidean_distance


class DustParticle(pygame.sprite.Sprite):
    def __init__(self,
                 position: tuple,
                 size: float,
                 surface_length: tuple,
                 i: int
                 ):
        pygame.sprite.Sprite.__init__(self)
        self.pos_x,self.pos_y = position
        self.radius = size
        self.i = i
        self.image = pygame.Surface(surface_length)
        self.image.set_colorkey((2, 2, 2))
        self.image.fill([2,2,2])
        self.rect = self.image.get_rect(topleft = (0,0))
        pygame.draw.circle(self.image, (0,0,0), (self.pos_x, self.pos_y), self.radius)

class Dust:
    def __init__(self,
                 dust_density: tuple,
                 dust_size: float,
                 surface_size:tuple
                 ):
        surface_x,surface_y = surface_size
        dust_density_x,dust_density_y = dust_density
        dust_x_coordinates = [i for i in range(120,surface_x-100,dust_density_x)]
        dust_y_coordinates = [i for i in range(120,surface_y-100,dust_density_y)]
        self.group = pygame.sprite.Group()
        index = 0
        for x in dust_x_coordinates:
            x_ = np.full(len(dust_x_coordinates),x)
            for pos in zip(x_,dust_y_coordinates):
                self.group.add(DustParticle(pos,dust_size,surface_size,index))
                index += 1

    def draw(self,
             surface: Surface
             ):
        self.group.draw(surface)
    
    def update(self,
             surface: Surface,
             agent
             ):
        for sprite in self.group:
            dist = euclidean_distance(agent.pos_x,agent.pos_y,sprite.pos_x,sprite.pos_y)
            if dist <= agent.radius:
                sprite.kill()
        self.group.draw(surface)


