import noise
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

shape = (1024,1024)
scale = 100.
octaves = 6
persistence = 0.5
lacunarity = 2.0
RANGE_MAX = 1


blue = np.array([65,105,225])
green = np.array([34,139,34])
beach = np.array([238, 214, 175])

def add_color(world):
    color_world = np.zeros(world.shape+(3,))
    for i in range(shape[0]):
        for j in range(shape[1]):
            if world[i][j] < -0.05:
                color_world[i][j] = blue
            elif world[i][j] < 0:
                color_world[i][j] = beach
            elif world[i][j] < 1.0:
                color_world[i][j] = green

    return color_world

world = np.zeros(shape)
for i in range(shape[0]):
    for j in range(shape[1]):
        world[i][j] = noise.pnoise2(i/scale, 
                                    j/scale, 
                                    octaves=octaves, 
                                    persistence=persistence, 
                                    lacunarity=lacunarity, 
                                    repeatx=1024, 
                                    repeaty=1024, 
                                    base=0)

color_world  = add_color(world)
formatted_world = (world * 255 / np.max(world)).astype('uint8')
image = Image.fromarray(formatted_world).save('./noise_base.png')
image_c = Image.fromarray(color_world, mode="RGB").save('./nose_color.png')