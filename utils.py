import cv2
import os
import numpy as np
def save_segmented_images(segmented_images, directory):
    """
    Save segmented images to a directory.

    :param segmented_images: List of tuples (image_path, segmented_image).
    :param directory: Directory to save the segmented images.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    for image_path, segmented_image in segmented_images:
        # Extract filename and create a full path
        filename = os.path.basename(image_path)
        save_path = os.path.join(directory, filename)

        # Save the segmented image
        cv2.imwrite(save_path, segmented_image)
def load_image(path, mode=cv2.IMREAD_COLOR):
    """
    Load an image from the given path.
    :param path: Path to the image file
    :param mode: Mode in which to read the image (default is grayscale)
    :return: Loaded image
    """
    return cv2.imread(path, mode)

def save_image(path, image):
    """
    Save an image to the given path.
    :param path: Path where to save the image
    :param image: Image to save
    """
    cv2.imwrite(path, image)

def dice_coefficient(im1, im2):
    """
    Compute the Dice coefficient between two images.
    :param im1: First image
    :param im2: Second image
    :return: Dice coefficient
    """
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / (im1.sum() + im2.sum())

def jaccard_index(im1, im2):
    """
    Compute the Jaccard index between two images.
    :param im1: First image
    :param im2: Second image
    :return: Jaccard index
    """
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    intersection = np.logical_and(im1, im2)
    union = np.logical_or(im1, im2)
    return intersection.sum() / float(union.sum())

def intersection_over_union(im1, im2):
    """
    Compute the Intersection over Union (IoU) between two images.
    :param im1: First image
    :param im2: Second image
    :return: IoU
    """
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    intersection = np.logical_and(im1, im2)
    union = np.logical_or(im1, im2)
    return intersection.sum() / float(union.sum())
def f1_score(segmented_image, ground_truth_image):
    """
    Calculate the F1 Score between the segmented image and the ground truth image.
    
    Parameters:
        segmented_image (numpy array): The image obtained after segmentation.
        ground_truth_image (numpy array): The ground truth image.
        
    Returns:
        float: The F1 Score.
    """
    # Convert the images to boolean arrays
    segmented_image_bool = np.asarray(segmented_image).astype(bool)
    ground_truth_image_bool = np.asarray(ground_truth_image).astype(bool)
    
    # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
    TP = np.sum(np.logical_and(segmented_image_bool, ground_truth_image_bool))
    FP = np.sum(np.logical_and(segmented_image_bool, np.logical_not(ground_truth_image_bool)))
    FN = np.sum(np.logical_and(np.logical_not(segmented_image_bool), ground_truth_image_bool))
    
    # Calculate Precision and Recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # Calculate F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1


