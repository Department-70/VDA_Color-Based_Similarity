import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


image_root = './images/'
gt_root = './GT/'
results_dir = './results'
counter = 0

# Load an image
for files in os.scandir(image_root):
    file_name = os.path.splitext(files.name)[0]
    image = cv2.imread(image_root + file_name + '.jpg')

    # Convert the image to the Lab color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab) # Calculate color gradients
    gradient_x = cv2.Sobel(lab_image[:, :, 1], cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(lab_image[:, :, 1], cv2.CV_64F, 0, 1, ksize=3) # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2) # Threshold the gradient magnitude to find subtle patterns
    threshold = 15  # Adjust this threshold based on your requirements
    subtle_pattern_mask = gradient_magnitude > threshold # Apply the mask to the original image
    subtle_pattern = image.copy()
    subtle_pattern[~subtle_pattern_mask] = [0, 0, 0]  # Set non-pattern areas to black
    
    # Save or display the result
    fig, axs = plt.subplots(1, 2, figsize=(10,5))
    
    axs[0].imshow(image)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(subtle_pattern)
    axs[1].set_title('Subtle Pattern Image')
    axs[1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join('results', 'results_' + file_name + '.jpg'))
    plt.close()

    counter += 1

    if counter >= 6000: #6000
        break