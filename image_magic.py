import matplotlib.pyplot as plt
import numpy as np
from perlin_noise import PerlinNoise as PN
from PIL import Image
import scipy.ndimage as filter

# Make an image consisting of perlin noise
def make_a_noise():
    parx = 500
    pary = 500
    noise = PN(octaves=5, seed=1)
    pic = np.array([[noise([i/parx, j/pary]) * 0.1 for j in range(pary)] for i in range(parx)])

    print(pic)
    plt.imshow(pic, cmap='gray')
    plt.show()

    return pic

# run a fast fourier transform on an image of noise, and return the real part of that transformed data
def make_a_better_noise(noise):
    conoise = np.fft.fft2(noise)
    conoise = np.fft.fftshift(conoise)

    print(conoise)
    plt.imshow(conoise.real, cmap='gray')
    plt.show()

    return conoise

def silence(noise):
    blurred = filter.gaussian_filter(noise, sigma=7)

    print(blurred)
    plt.imshow(blurred, cmap='gray')
    plt.show()

def make_a_picture(noise, filename):

    image = Image.fromarray(noise, 'F')
    image.show()
    image.save(filename)


image_root = './images/'
pic = make_a_noise()
pic2 = make_a_better_noise(pic)
pic3 = silence(pic)

# Notes
# Look into running an FFT on the color map, or even the image, 
# fft brings things into the frequency domain
# talk to alex for more info