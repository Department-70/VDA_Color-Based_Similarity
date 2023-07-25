# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 14:53:52 2023

@author: Debra Hogue

UMAP reduction of images to use GMM for object detection?
"""

import cv2
from matplotlib.colors import ListedColormap, Normalize
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image, ImageFilter
# from scipy.ndimage.filters import sobel
from scipy.ndimage import binary_dilation
from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
import time
import umap

image_root = './images/'
gt_root = './GT/'

"""
===================================================================================================
    Helper function
        - Detect an object with UMAP and GMM
        
        Multidimensional space analysis - human-in-the-loop
===================================================================================================
"""
def detect_object(image_matrix):
    # To guarantee reproducable results
    random_state=42
    
    # Define the UMAP parameters
    n_neighbors = 15
    n_components = 2

    # Create a UMAP object and fit_transform the data
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, random_state=random_state)
    embedding = reducer.fit_transform(image_matrix)

    # Define the number of clusters
    n_clusters = 3

    # Create a GMM object
    gmm = GaussianMixture(n_components=n_clusters,random_state=random_state)

    # Fit the GMM to the embedding data
    gmm.fit(embedding)  # embedding

    # Obtain the probabilities of cluster assignments
    probs = gmm.predict_proba(embedding)

    # Define the probability threshold
    threshold = 0.5  # Set the desired threshold

    # Determine if an object is present based on the threshold
    object_indices = np.where(probs[:, 0] > threshold)[0]  # Cluster 0 probabilities

    if len(object_indices) > 0:
        print("An object is present in the image.")
        return True
    else:
        print("No object is present in the image.")
        return False


"""
===================================================================================================
    Step 1: Use K-Means to cluster the similar colors present in the background and foreground
===================================================================================================
"""
def color_heatmap(image, num_clusters=None):
    # Convert the image to grayscale
    gray_image = image.convert("L")

    # Apply Gaussian smoothing to reduce noise
    smoothed_image = gray_image.filter(ImageFilter.GaussianBlur(radius=3))

    # Reshape the image to a 2D array of pixels
    pixels = np.array(smoothed_image).reshape(-1, 1)

    # Determine the number of clusters if not provided
    if num_clusters is None:
        num_clusters = int(np.sqrt(pixels.shape[0] / 2))
        num_clusters = max(min(num_clusters, 5), 2)
        print('num_clusters = ' + str(num_clusters))

    # Apply K-means clustering to segment the image based on color
    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(pixels)

    # Create a custom colormap for the cluster labels
    cmap = create_custom_colormap(num_clusters)

    # Reshape the labels to match the original image shape
    segmented = labels.reshape(gray_image.size[1], gray_image.size[0])

    # Create a heatmap based on the cluster labels
    heatmap = cmap(segmented / (num_clusters - 1))

    return heatmap, num_clusters

"""
===================================================================================================
    Step 2: Use the ground truth to segment out the object from the heatmap
===================================================================================================
"""
def add_outline_to_heatmap(color_heatmap, binary_image, outline_thickness=5):
    # Convert the binary image to boolean
    binary_image = binary_image.astype(bool)

    # Perform binary dilation on the binary image
    dilated_image = binary_dilation(binary_image, iterations=outline_thickness)

    # Create a copy of the color heatmap
    outlined_heatmap = color_heatmap.copy()

    # Get the alpha channel from the color heatmap
    alpha_channel = outlined_heatmap[:, :, 3]

    # Set the outline pixels to white (1.0, 1.0, 1.0) with alpha = 1.0
    outlined_heatmap[dilated_image, :3] = [1.0, 1.0, 1.0]
    outlined_heatmap[dilated_image, 3] = alpha_channel[dilated_image]

    return outlined_heatmap

"""
===================================================================================================
    Step 3: Replace the white cutout in the new heatmap with a photorealistic render of the 
            object with the segmented colors from the first heatmap
===================================================================================================
"""
def render_object_with_colors(file_name, image, color_heatmap, binary_image):    
    # Convert the binary image to boolean
    mask = binary_image.astype(bool)

    # Create a copy of the color_heatmap
    rendered_image = color_heatmap.copy()

    # Ensure both image and rendered_image have the same number of channels
    if rendered_image.shape[2] == 4 and image.shape[2] == 3:
        image = np.concatenate([image, np.ones_like(image[..., :1])], axis=2)

    # Loop over the mask's pixels and set the corresponding pixels in rendered_image
    for i in range(rendered_image.shape[0]):
        for j in range(rendered_image.shape[1]):
            if mask[i, j]:
                rendered_image[i, j] = image[i, j]
                # print('image_value = ' + str(image[i, j]))
            else:
                rendered_image[i, j] = color_heatmap[i, j] * 255
                # print('color_heatmap = ' + str(color_heatmap[i, j] * 255))

    # Convert the rendered_image to unsigned integer data type
    rendered_image = (rendered_image).astype(np.uint8)
    
    # Save the rendered_image as an HD image
    cv2.imwrite('./color_gestalt_output/' + file_name + '.png', rendered_image)

    return rendered_image

def draw_diagonal_lines_on_background(blended_image, binary_image, line_size=20):
    # Convert the binary image to boolean
    mask = binary_image.astype(bool)

    # Copy the blended_image
    output_image = blended_image.copy()

    # Create a diagonal lines pattern with transparency
    line_pattern = np.ones((line_size, line_size, 4), dtype=np.uint8) * 255

    # Set the diagonal lines to 0
    for i in range(line_size):
        line_pattern[i, i] = [0, 0, 0, 0]

    # Iterate over the binary image and draw diagonal lines where black pixels are present
    for i in range(output_image.shape[0]):
        for j in range(output_image.shape[1]):
            if not mask[i, j]:
                # Calculate the region to place the diagonal lines centered at the current black pixel
                line_start_i = max(0, i - line_size // 2)
                line_start_j = max(0, j - line_size // 2)
                line_end_i = min(output_image.shape[0], line_start_i + line_size)
                line_end_j = min(output_image.shape[1], line_start_j + line_size)

                # Adjust the dimensions of the diagonal lines pattern to fit within the region
                line_pattern_height = line_end_i - line_start_i
                line_pattern_width = line_end_j - line_start_j
                line_pattern_adjusted = line_pattern[:line_pattern_height, :line_pattern_width]

                # Draw the diagonal lines pattern at the current position
                output_image[line_start_i:line_end_i, line_start_j:line_end_j] = line_pattern_adjusted

    # Convert the rendered_image to unsigned integer data type
    output_image = (output_image).astype(np.uint8)

    return output_image

"""
===================================================================================================
    Helper Function 
        Creates a custom bivariate color map (Green to Purple)
===================================================================================================
"""
def create_custom_colormap(num_clusters):
    green = np.array([0, 0.4, 0])  # Green color
    purple = np.array([0.9, 0, 0.9])  # Purple color
    colors = [green + (purple - green) * i / (num_clusters - 1) for i in range(num_clusters)]
    cmap = ListedColormap(colors)

    return cmap

def apply_mask_with_alpha(image_array, mask_array, alpha=0.7):
    # Ensure both image and rendered_image have the same number of channels
    if mask_array.shape[2] == 4 and image_array.shape[2] == 3:
        image_array = np.concatenate([image_array, np.ones_like(image_array[..., :1])], axis=2)

    # Create an alpha channel for the mask
    mask_alpha = np.ones_like(mask_array) * alpha

    # Apply alpha blending
    blended_image = (image_array * (1 - mask_alpha) + mask_array * mask_alpha * 255).astype(np.uint8)

    # Set the alpha channel of the blended image to 255
    blended_image[..., 3] = 255

    return blended_image

"""
===================================================================================================
    Main
===================================================================================================
"""
if __name__ == "__main__":
    # Counter
    counter = 1
        
    # Loop to iterate through dataset
    for files in os.scandir(image_root):
        print("Counter = " + str(counter) + '.')
        
        # Filename
        file_name = os.path.splitext(files.name)[0]
    
        # Read in Image & Display
        image = Image.open(image_root + file_name + '.jpg')
        gt = Image.open(gt_root + file_name + '.png')
        
        # Convert image into a NumPy array
        image_array = np.array(image)
        
        # Reshape the image array to a 2D matrix
        image_matrix = image_array.reshape(image_array.shape[0], -1)
        
        # Add alpha channel to image_array
        image_array = np.concatenate([image_array, np.ones_like(image_array[..., :1])], axis=2)
        image_array[..., 3] = 255
        
        binary_image_array = np.array(gt)
       
                
        # UMAP and GMM for object detection
        object_present = detect_object(image_matrix)
        
        if object_present:
            # Start the timer
            start_time = time.time()
            
            # num_clusters = 5
            heatmap, num_clusters = color_heatmap(image)
            
            # Calculate the elapsed time
            elapsed_time = time.time() - start_time
            
            # Print the elapsed time
            print("Elapsed time:", elapsed_time, "seconds")
            
            # Assuming you have the color heatmap numpy array and the binary image numpy array
            # outlined_heatmap = add_outline_to_heatmap(heatmap, binary_image_array)
            
            # Apply the colormap as a mask onto the original image
            masked_image = apply_mask_with_alpha(image_array, heatmap)
            
            # result = draw_diagonal_lines_on_background(masked_image, binary_image_array)
            
            # Assuming you have the image, color_heatmap, outlined_heatmap, and binary_image arrays
            rendered_image = render_object_with_colors(file_name, masked_image, heatmap, binary_image_array)
                    
            
            # Create a side-by-side plot of the original image and the heatmap
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(image)
            axs[0].set_title("Original Image")
            axs[0].axis('off')
            
            axs[1].imshow(rendered_image)
            axs[1].set_title("Color Heatmap ({} Clusters)".format(num_clusters))
            axs[1].axis('off')
            
            plt.tight_layout()
                
            # plt.title('Color-Based Gestalt Similarity')
            plt.savefig("results/results_" + file_name + 'jpg')
            plt.show()
            
            
            
            # Take the heatmap and apply a Sobel filter
            # Calculate multiple independant clusters
            # Evaluate results
            
        counter += 1
        
        if counter == 6001:
            break