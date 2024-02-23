import os
import numpy as np
from PIL import Image, ImageDraw


def generate_circle(image_size):
    image = Image.new("RGB", image_size)
    draw = ImageDraw.Draw(image)
    center = (np.random.randint(0, image_size[0]), np.random.randint(0, image_size[1]))
    radius = np.random.randint(10, min(image_size) // 4)  # Random radius
    draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), fill="white")
    return image


def generate_square(image_size):
    image = Image.new("RGB", image_size)
    draw = ImageDraw.Draw(image)
    upper_left = (np.random.randint(0, image_size[0]), np.random.randint(0, image_size[1]))
    side_length = np.random.randint(10, min(image_size) // 2)  # Random side length
    lower_right = (upper_left[0] + side_length, upper_left[1] + side_length)
    draw.rectangle([upper_left, lower_right], fill="white")
    return image

def generate_images(num_images_per_class, image_size):
    images = []
    for i in range(num_images_per_class):
        circle_image = generate_circle(image_size)
        images.append(circle_image)

        square_image = generate_square(image_size)
        images.append(square_image)
    return images


# Modify the main function to accept directory paths for circles and squares
def main(num_images_per_class, image_size, circle_dir, square_dir):
    for i in range(num_images_per_class):
        circle_image = generate_circle(image_size)
        circle_image.save(os.path.join(circle_dir, f"circle_{i}.png"))

        square_image = generate_square(image_size)
        square_image.save(os.path.join(square_dir, f"square_{i}.png"))

if __name__ == "__main__":
    num_images_per_class = 2000
    image_size = (255, 255)  # Set the size of the images
    circle_dir = "/home/james/projects/circlesquareai-master/circle"  # Specify the directory to save circle images
    square_dir = "/home/james/projects/circlesquareai-master/square"  # Specify the directory to save square images
    os.makedirs(circle_dir, exist_ok=True)
    os.makedirs(square_dir, exist_ok=True)
    main(num_images_per_class, image_size, circle_dir, square_dir)




