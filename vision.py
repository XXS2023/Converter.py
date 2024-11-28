from PIL import Image, ImageFilter
import cv2
import numpy as np
import os

def save_character_contours(image_path, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to get a binary image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right based on x-coordinate
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    # Process each contour
    for i, contour in enumerate(contours):
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Create a square bounding box
        size = max(w, h)
        square_x = x + w // 2 - size // 2
        square_y = y + h // 2 - size // 2

        # Crop the character from the original image
        character_image = image[square_y:square_y + size, square_x:square_x + size]

        # Save the character image as JPG file
        output_path = os.path.join(output_folder, f'character_{i}.jpg')
        cv2.imwrite(output_path, character_image)

# Example usage

save_character_contours('pronaun.jpg', 'pronaun_characters')

