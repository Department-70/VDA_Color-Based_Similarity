# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 14:53:52 2023

@authors: Debra Hogue, Zak Kastl

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
    pixels = np.array(gray_image).reshape(-1, 1)

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

    return heatmap, num_clusters, cmap

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
def apply_ordered_dithering(values, dither_matrix):
    def dither_element(value):
        return 255 if value > dither_matrix else 0

    dither_element_vectorized = np.vectorize(dither_element)
    return dither_element_vectorized(values)

def render_object_with_colors(file_name, image, color_heatmap, binary_image):    
    # Convert the binary image to boolean
    mask = binary_image.astype(bool)

    # Create a copy of the color_heatmap
    rendered_image = color_heatmap.copy()

    # Ensure both image and rendered_image have the same number of channels
    if rendered_image.shape[2] == 4 and image.shape[2] == 3:
        image = np.concatenate([image, np.ones_like(image[..., :1])], axis=2)
        
    # Define ordered dithering matrices for different colors
    dither_matrix_green = np.array([[0, 128, 32, 160],
                                    [192, 64, 224, 96],
                                    [48, 176, 16, 144],
                                    [240, 112, 208, 80]])

    dither_matrix_purple = np.array([[192, 64, 224, 96],
                                     [48, 176, 16, 144],
                                     [240, 112, 208, 80],
                                     [0, 128, 32, 160]])

    # Loop over the mask's pixels and set the corresponding pixels in rendered_image
    for i in range(rendered_image.shape[0]):
        for j in range(rendered_image.shape[1]):
            if mask[i, j]:
                rendered_image[i, j] = image[i, j]
                # print('image_value = ' + str(image[i, j]))
            else:
                # Get the color heatmap value for the current pixel
                color_value = color_heatmap[i, j] * 255

                # Determine which cluster the pixel belongs to
                cluster = np.argmax(color_heatmap[i, j])

                # Apply the appropriate dithering based on the cluster color
                if cluster == 0:
                    dither_value = dither_matrix_green[i % 4, j % 4]
                else:
                    dither_value = dither_matrix_purple[i % 4, j % 4]

                rendered_image[i, j] = apply_ordered_dithering(color_value, dither_value)
                #rendered_image[i, j] = color_heatmap[i, j] * 255
                # print('color_heatmap = ' + str(color_heatmap[i, j] * 255))

    # Convert the rendered_image to unsigned integer data type
    rendered_image = (rendered_image).astype(np.uint8)
    
    # Save the rendered_image as an HD image
    cv2.imwrite('./color_gestalt_output/' + file_name + '.png', rendered_image)

    return rendered_image


"""
===================================================================================================
    Step 4: Calculate the 'camo-ness' of the heat map against the ground truth.
    This could be 
===================================================================================================
"""
def create_histogram(heatmap, color_list: ListedColormap, ground_truth):
    ## 1. Add alpha channel to the color map for comparison purposes
    ## 2. compare heatmap color pixel-by-pixel (ew) and organize by color?
    ### 2.1. Create dictionary for histogram to organize the following information:
    ###     - Color Name: Key
    ###     - RGBA value as list
    ###     - Number of pixels of that color
    ###     - Inside/Outside (do later)

    # reshape the arrays into matching sizes
    pixels = heatmap.reshape(np.shape(heatmap)[0]*np.shape(heatmap)[1], np.shape(heatmap)[2])
    gt = ground_truth.reshape(np.shape(ground_truth)[0]*np.shape(ground_truth)[1])
    colors = []
    for v in color_list.colors:
        vv = v.tolist()
        vv.append(1.)
        colors.append(vv)

    colors = np.array(colors)
    out_buckets = [0] * len(colors)
    in_buckets = [0]*len(colors)

    # update the color map to 

    for i in range(len(pixels)):
        for j in range(len(colors)):
            if np.all(colors[j] == pixels[i]):
                if(gt[i] == 255):
                    in_buckets[j] += 1
                else:
                    out_buckets[j] += 1

    return out_buckets, in_buckets
"""
===================================================================================================
    Step 5: Evaluate the color distributions & patterns
===================================================================================================
"""
import mahotas.features

def evaluate_background_patterns(rendered_image, binary_mask):
    # Convert the rendered_image to grayscale
    gray_image = np.dot(rendered_image[...,:3], [0.2989, 0.587, 0.114]).astype(np.uint8)

    # Calculate the LBP image for the grayscale image
    lbp_image = mahotas.features.lbp(gray_image, radius=1, points=8)

    # Convert the binary_mask to a NumPy array and then to boolean
    mask = np.array(binary_mask).astype(bool)

    # Calculate Haralick texture features for the entire LBP image
    haralick_features = mahotas.features.haralick(lbp_image)

    # Get the pixels from the LBP image corresponding to the background
    background_pixels = haralick_features[~mask]

    # Calculate the average Haralick texture features for the background
    average_haralick_features = np.mean(background_pixels, axis=0)

    return average_haralick_features


def evaluate_color_distribution(filename, colormap, rendered_image, binary_map):
    # Calculate the mean color of the colormap (without alpha channel)
    mean_colormap_color = np.mean(colormap[..., :3], axis=0)
 
    # Calculate the mean color of the color heatmap (without alpha channel)
    mean_color_heatmap = np.mean(rendered_image[..., :3], axis=(0, 1))
 
    # Calculate the Euclidean distance between the mean colormap color and mean color heatmap
    euclidean_distance = np.linalg.norm(mean_colormap_color - mean_color_heatmap)
 
    # Create a mask to select pixels where the object is present
    object_mask = np.logical_not(binary_map)
 
    # Calculate the mean color for ground (black) and figure (white) regions
    ground_color = np.mean(rendered_image[object_mask], axis=0)
    figure_color = np.mean(rendered_image[np.logical_not(object_mask)], axis=0)
 
    # Calculate the standard deviation for ground and figure regions
    ground_std_dev = np.std(rendered_image[object_mask], axis=0)
    figure_std_dev = np.std(rendered_image[np.logical_not(object_mask)], axis=0)
 
    # Calculate the absolute difference in standard deviations
    std_dev_diff = np.abs(ground_std_dev - figure_std_dev)
 
    # Calculate the mean absolute error (MAE) between the colormap and rendered image (without alpha channel)
    mae = np.mean(np.abs(colormap[..., :3] - rendered_image[..., :3]), axis=(0, 1))
 
    # Calculate the root mean squared error (RMSE) between the colormap and rendered image (without alpha channel)
    rmse = np.sqrt(np.mean((colormap[..., :3] - rendered_image[..., :3]) ** 2, axis=(0, 1)))
 
    # Get the green and purple color components for the figure and the ground
    green_values_figure = rendered_image[np.logical_not(object_mask)][:, 1]
    purple_values_figure = rendered_image[np.logical_not(object_mask)][:, 2]
 
    green_values_ground = rendered_image[object_mask][:, 1]
    purple_values_ground = rendered_image[object_mask][:, 2]
 
    # Plot histograms for green and purple color components in the figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].hist(green_values_figure.flatten(), bins=50, alpha=0.7, color='green', label='Green (Figure)')
    axes[0, 0].set_title('Green Color Component Histogram (Figure)')
    axes[0, 0].set_xlabel('Green Color Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
 
    axes[0, 1].hist(purple_values_figure.flatten(), bins=50, alpha=0.7, color='purple', label='Purple (Figure)')
    axes[0, 1].set_title('Purple Color Component Histogram (Figure)')
    axes[0, 1].set_xlabel('Purple Color Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
 
    # Plot histograms for green and purple color components in the ground
    axes[1, 0].hist(green_values_ground.flatten(), bins=50, alpha=0.7, color='green', label='Green (Ground)')
    axes[1, 0].set_title('Green Color Component Histogram (Ground)')
    axes[1, 0].set_xlabel('Green Color Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
 
    axes[1, 1].hist(purple_values_ground.flatten(), bins=50, alpha=0.7, color='purple', label='Purple (Ground)')
    axes[1, 1].set_title('Purple Color Component Histogram (Ground)')
    axes[1, 1].set_xlabel('Purple Color Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
 
    # Save the histograms as an image
    plt.savefig(f'./evaluation_results/{filename}_histograms.png')
    plt.close()
 
    # Save the evaluation results to a .txt file
    with open('./evaluation_results/' + filename + '_evaluation_results.txt', 'w') as file:
         file.write(f"Mean Colormap Color: {mean_colormap_color}\n")
         file.write(f"Mean Color Heatmap: {mean_color_heatmap}\n")
         file.write(f"Euclidean Distance: {euclidean_distance}\n")
         file.write(f"Ground Color: {ground_color}\n")
         file.write(f"Figure Color: {figure_color}\n")
         file.write(f"Ground Std Dev: {ground_std_dev}\n")
         file.write(f"Figure Std Dev: {figure_std_dev}\n")
         file.write(f"Std Dev Difference: {std_dev_diff}\n")
         file.write(f"Mean Absolute Error (MAE): {mae}\n")
         file.write(f"Root Mean Squared Error (RMSE): {rmse}\n")
         
    # Return the evaluation metrics as a dictionary
    evaluation_results = {
         'mean_colormap_color': mean_colormap_color,
         'mean_color_heatmap': mean_color_heatmap,
         'euclidean_distance': euclidean_distance,
         'ground_color': ground_color,
         'figure_color': figure_color,
         'ground_std_dev': ground_std_dev,
         'figure_std_dev': figure_std_dev,
         'std_dev_diff': std_dev_diff,
         'mae': mae,
         'rmse': rmse
    }
 
    return evaluation_results

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
    percent_purple = []
        
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
        image_array[..., 3] = 0
        
        binary_image_array = np.array(gt)
       
                
        # UMAP and GMM for object detection
        object_present = detect_object(image_matrix)
        
        if object_present:
            # Start the timer
            start_time = time.time()
            
            # num_clusters = 5
            heatmap, num_clusters, cmap = color_heatmap(image)

            # NEW CODE
            color_distro_out, color_distro_in = create_histogram(heatmap, cmap, np.array(gt))
            color_normal_out = [i/sum(color_distro_out) for i in color_distro_out]
            color_normal_in = [i/sum(color_distro_in) for i in color_distro_in]
            percent_purple.append((color_normal_out[4], color_normal_in[4]))
            
            # END NEW CODE
            
            # Calculate the elapsed time
            elapsed_time = time.time() - start_time
            
            # Print the elapsed time
            print("Elapsed time:", elapsed_time, "seconds")
                         
            # Apply the colormap as a mask onto the original image
            masked_image = apply_mask_with_alpha(image_array, heatmap)
            
            # Assuming you have the image, color_heatmap, outlined_heatmap, and binary_image arrays
            rendered_image = render_object_with_colors(file_name, masked_image, heatmap, binary_image_array)
            
            # Create a side-by-side plot of the original image and the heatmap
            fig, axs = plt.subplots(figsize=(10, 5))

            '''axs[0][0].imshow(image)
            axs[0][0].set_title("Original Image")
            axs[0][0].axis('off')
            
            axs[0][1].imshow(rendered_image)
            axs[0][1].set_title("Color Heatmap ({} Clusters)".format(num_clusters))
            axs[0][1].axis('off')

            axs[0][2].imshow(heatmap)
            axs[0][2].set_title("Raw Heatmap")
            axs[0][2].axis("off")

            axs[1][0].imshow(masked_image)
            axs[1][0].set_title("Masked Image")
            axs[1][0].axis("off")'''
            
            catagories = ['One', 'Two', 'Three', 'Four', 'Five']
            axs.bar(catagories, color_normal_out, width=0.2, color=(cmap.colors), align='center')
            axs.bar(catagories, color_normal_in, width=0.2, color='black', align='edge')
            
            axs.set_title('Distribtuion of colors (Figure vs Ground)')
            
            plt.tight_layout()
                
            #plt.title('Color-Based Gestalt Similarity')
            plt.savefig('results/results_' + file_name + '.jpg')
            #plt.show()

            # Take the heatmap and apply a Sobel filter
            # Calculate multiple independent clusters
            # Evaluate results
            evaluation_results = evaluate_color_distribution(file_name, heatmap, rendered_image, gt)
            print(evaluation_results)
            print(percent_purple)
        
            # ZAK - Have we considered putting the heatmap through an 'anti-perlin noise filter'? For example,
            # Randomness should produce perlin noise, the inverse should produce nothing?
            
        counter += 1
        
        if counter == 6001: #6001:
            break
