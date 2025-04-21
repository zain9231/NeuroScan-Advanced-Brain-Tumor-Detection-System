import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
import os
import tkinter as tk
from tkinter import filedialog
import shutil


def convert_to_rgb(image):
    """Convert grayscale or RGBA images to RGB."""
    if len(image.shape) == 2:  # Grayscale
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:  # RGBA
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return image  # Already RGB

def safe_rotate_image(image, angle):
    """Rotate image safely with proper handling."""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def safe_translate_image(image, dx, dy):
    """Translate image safely."""
    height, width = image.shape[:2]
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, translation_matrix, (width, height), borderMode=cv2.BORDER_REFLECT)

def flip_image(image, direction):
    """Flip image based on the direction."""
    if direction == 'horizontal':
        return cv2.flip(image, 1)
    elif direction == 'vertical':
        return cv2.flip(image, 0)
    return image

def add_noise(image, noise_type='gaussian', amount=0.8):
    """Add noise to image."""
    row, col, ch = image.shape
    mean = 0
    var = amount * 255
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def scale_image(image, scale_factor):
    """Scale image by the scale factor."""
    return cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

def elastic_transform(image, alpha, sigma, random_state=None):
    """Apply elastic transformation."""
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape[:2]) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    distorted_image = np.zeros_like(image)
    for i in range(image.shape[2]):  # Loop through each channel
        distorted_image[:, :, i] = map_coordinates(image[:, :, i], indices, order=1, mode='reflect').reshape(shape[:2])

    return distorted_image

def adjust_intensity(image, factor):
    """Adjust image intensity."""
    return np.clip(image * factor, 0, 255).astype(np.uint8)

def shear_image(image, shear_factor):
    """Apply shearing to image."""
    rows, cols = image.shape[:2]
    shear_matrix = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    return cv2.warpAffine(image, shear_matrix, (cols, rows))

def random_crop(image, crop_height, crop_width):
    """Randomly crop the image."""
    height, width = image.shape[:2]
    if height < crop_height or width < crop_width:
        return image
    
    y = np.random.randint(0, height - crop_height)
    x = np.random.randint(0, width - crop_width)
    return image[y:y+crop_height, x:x+crop_width]

def augment_image(image, augment_options):
    """Apply augmentations based on options."""
    augmented_images = []
    image = convert_to_rgb(image)  # Ensure image is RGB

    if augment_options.get("rotation", False):
        rotated = safe_rotate_image(image, np.random.uniform(-30, 30))
        augmented_images.append(("rotation", rotated))

    if augment_options.get("translation", False):
        translated = safe_translate_image(image, np.random.uniform(-50, 50), np.random.uniform(-50, 50))
        augmented_images.append(("translation", translated))

    if augment_options.get("scaling", False):
        scaled = scale_image(image, np.random.uniform(0.8, 1.2))
        augmented_images.append(("scaling", scaled))

    if augment_options.get("flipping", False):
        flipped = flip_image(image, np.random.choice(['horizontal', 'vertical']))
        augmented_images.append(("flipping", flipped))

    if augment_options.get("elastic_deformation", False):
        elastic = elastic_transform(image, alpha=1000, sigma=40)
        augmented_images.append(("elastic_deformation", elastic))

    if augment_options.get("intensity_adjustment", False):
        adjusted = adjust_intensity(image, np.random.uniform(0.8, 1.2))
        augmented_images.append(("intensity_adjustment", adjusted))

    if augment_options.get("noise_injection", False):
        noisy = add_noise(image, 'gaussian', amount=np.random.uniform(0.5, 0.8))
        augmented_images.append(("noise_injection", noisy))

    if augment_options.get("shearing", False):
        sheared = shear_image(image, np.random.uniform(-0.2, 0.2))
        augmented_images.append(("shearing", sheared))

    if augment_options.get("random_cropping", False):
        crop_size = int(min(image.shape[:2]) * 0.8)
        cropped = random_crop(image, crop_size, crop_size)
        cropped = cv2.resize(cropped, (image.shape[1], image.shape[0]))
        augmented_images.append(("random_cropping", cropped))

    return augmented_images

def augment_dataset(input_folder, output_folder, augment_options, num_augmented_images, progress_callback=None):
    """Process dataset and save augmented images."""
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    processed_images = 0
    total_images = len([f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for i, filename in enumerate(os.listdir(input_folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Unable to read image: {image_path}")
                continue

            augmented_images = augment_image(image, augment_options)
            for j in range(num_augmented_images):
                for technique, aug_image in augmented_images:
                    output_filename = f"{technique}_{j}_{filename}"
                    output_path = os.path.join(output_folder, output_filename)
                    cv2.imwrite(output_path, aug_image)
            
            processed_images += 1
            
            if progress_callback:
                progress = (i + 1) / total_images * 100
                progress_callback(progress)

    return processed_images

if __name__ == "__main__":
    augment_options = {
        "rotation": True,
        "translation": True,
        "scaling": True,
        "flipping": True,
        "elastic_deformation": True,
        "intensity_adjustment": True,
        "noise_injection": True,
        "shearing": True,
        "random_cropping": True
    }
    output_folder = "augmented_output"
    #num_augmented_images = 5