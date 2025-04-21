import cv2
import numpy as np
from skimage import exposure

def normalize_image(image):
    """Normalize the image to 0-255 range."""
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def reduce_noise(image):
    """Apply Gaussian blur to reduce noise."""
    return cv2.GaussianBlur(image, (5, 5), 0)

def skull_strip(image):
    """
    Perform basic skull stripping using thresholding.
    
    """
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_and(image, image, mask=thresh)

def remove_artifacts(image):
    """
    Remove artifacts using adaptive histogram equalization.
    """
    return exposure.equalize_adapthist(image)

def preprocess_image(image_path, preprocess_options):
    """
    Preprocess the image based on the selected options.
    
    Args:
    image_path (str): Path to the input image.
    preprocess_options (dict): Dictionary of preprocessing options.
    
    Returns:
    numpy.ndarray: Preprocessed image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image at {image_path}")
    
    # If all preprocessing options are False, return the original image
    if not any(preprocess_options.values()):
        return image

    # Convert to grayscale if any preprocessing is applied
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if preprocess_options["image_normalization"]:
        image = normalize_image(image)
    
    if preprocess_options["noise_reduction"]:
        image = reduce_noise(image)
    
    if preprocess_options["skull_stripping"]:
        image = skull_strip(image)
    
    if preprocess_options["artifact_removal"]:
        image = remove_artifacts(image)
        # Convert image to uint8 after equalization to prevent black image
        image = (image * 255).astype(np.uint8)
    
    return image