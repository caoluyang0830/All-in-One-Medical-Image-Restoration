import os
import cv2
import numpy as np
from glob import glob
import random


def add_reflection_spots(image, num_spots=3, max_intensity=0.8, max_radius_ratio=0.1):
    """
    Add artificial reflection spots to a fundus image
    :param image: Input fundus image (BGR format)
    :param num_spots: Number of reflection spots to add
    :param max_intensity: Maximum intensity of spots (0-1)
    :param max_radius_ratio: Maximum radius of spots as ratio of image width
    :return: Degraded image with reflection spots
    """
    h, w = image.shape[:2]
    output = image.copy()

    for _ in range(num_spots):
        # Random parameters for each spot
        center_x = random.randint(int(w * 0.2), int(w * 0.8))
        center_y = random.randint(int(h * 0.2), int(h * 0.8))
        radius = random.randint(int(w * 0.01), int(w * max_radius_ratio))
        intensity = random.uniform(0.5, max_intensity)

        # Create a circular mask for the spot
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (center_x, center_y), radius, intensity, -1)

        # Apply Gaussian blur to soften the edges
        mask = cv2.GaussianBlur(mask, (0, 0), radius / 3)

        # Add the spot to all channels
        for c in range(3):
            output[:, :, c] = np.clip(output[:, :, c] + mask * 255, 0, 255)

    return output.astype(np.uint8)


def process_fundus_images(input_folder, output_folder):
    """
    Process all PNG fundus images in input folder and save degraded versions to output folder
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_paths = glob(os.path.join(input_folder, '*.png'))

    for img_path in image_paths:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Add reflection spots
        degraded_img = add_reflection_spots(img,
                                            num_spots=random.randint(2, 5),
                                            max_intensity=random.uniform(0.6, 0.9),
                                            max_radius_ratio=random.uniform(0.05, 0.15))

        # Save degraded image
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, degraded_img)

        print(f"Processed: {filename}")


if __name__ == "__main__":
    input_folder = "path/to/your/fundus_images"  # Replace with your input folder path
    output_folder = "path/to/output_folder"  # Replace with your desired output folder

    process_fundus_images(input_folder, output_folder)
    print("All images processed!")