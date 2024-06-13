import numpy as np
import os

image_folder = 'data/centreline_set/axial/npy/imgs'

images = []

# Load each image file
for filename in os.listdir(image_folder):
    if filename.endswith('.npy'):
        file_path = os.path.join(image_folder, filename)
        # Load the image file
        image = np.load(file_path)
        # Add to the list
        images.append(image)


all_images = np.stack(images)

mean = np.mean(all_images, axis=(0, 1, 2))
std = np.std(all_images, axis=(0, 1, 2))

print('Pixel Mean:', mean)
print('Pixel Standard Deviation:', std)
