import noise
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

shape = (1024,1024)
scale = 100.
octaves = 6
persistence = 0.5
lacunarity = 2.0
RANGE_MAX = 10

for map in range(RANGE_MAX):
    world = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            world[i][j] = noise.pnoise2(i/scale,
                                    j/scale,
                                    octaves=map + 1,
                                    persistence=persistence,
                                    lacunarity=lacunarity,
                                    repeatx=1024,
                                    repeaty=1024,
                                    base=0)

    formatted_world = (world * 255 / np.max(world)).astype('uint8')
    image = Image.fromarray(formatted_world).save('./noise_' + str(map) + '.png')