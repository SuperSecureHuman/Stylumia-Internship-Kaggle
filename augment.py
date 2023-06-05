import os
import random
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import shutil
import cv2
from tqdm import tqdm

# Set the paths and filenames
# Path to the folder containing images
data_folder = '/home/venom/repo/Stylumia-Internship-Kaggle/Dataset/train'
# Path to the CSV file containing image labels
csv_file = '/home/venom/repo/Stylumia-Internship-Kaggle/Dataset/train.csv'
# Path to the folder where augmented images will be saved
output_folder = '/home/venom/repo/Stylumia-Internship-Kaggle/Dataset/augmented_images'

# Define the augmentation transformations
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.Blur(p=0.2)
])


# Read the CSV file
df = pd.read_csv(csv_file)

# Calculate the desired number of images per class
target_num_images = 3000  # Modify this value as desired

# Count the number of images in each class
class_counts = df['label'].value_counts().to_dict()

# Iterate over each class
for label, count in class_counts.items():
    # Create the output folder for the current class
    output_label_folder = os.path.join(output_folder, str(label))
    os.makedirs(output_label_folder, exist_ok=True)

    # Check if augmentation is needed for the current class
    if count < target_num_images:
        # Calculate the number of additional images to generate
        num_augmented_images = target_num_images - count

        # Select all images from the current class for augmentation
        class_df = df[df['label'] == label]

        # Perform augmentation on the selected images
        augmented_images_count = 0
        while augmented_images_count < num_augmented_images:
            for _, row in class_df.iterrows():
                # Check if the desired number of augmented images is reached
                if augmented_images_count >= num_augmented_images:
                    break

                image_filename = row['file_name']

                # Load the image
                image_path = os.path.join(data_folder, image_filename)
                image = cv2.imread(image_path)
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Augment the image
                augmented = transform(image=image)
                augmented_image = augmented['image']

                # Save the augmented image
                output_filename = os.path.splitext(image_filename)[
                    0] + f'_augmented{augmented_images_count}.jpg'
                output_path = os.path.join(
                    output_label_folder, output_filename)
                cv2.imwrite(output_path, augmented_image)

                augmented_images_count += 1

        # Save the original images
        for _, row in tqdm(class_df.iterrows(), total=count, desc=f"Copying class {label}"):
            image_filename = row['file_name']
            image_path = os.path.join(data_folder, image_filename)
            original_output_path = os.path.join(
                output_label_folder, image_filename)
            shutil.copyfile(image_path, original_output_path)

    else:
        # Randomly sample 3000 images from the existing class
        class_df = df[df['label'] == label]
        random_indices = random.sample(range(count), target_num_images)
        random_images = class_df.iloc[random_indices]

        for _, row in tqdm(random_images.iterrows(), total=target_num_images, desc=f"Sampling class {label}"):
            image_filename = row['file_name']

            # Load the image
            image_path = os.path.join(data_folder, image_filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Save the original image
            original_output_path = os.path.join(
                output_label_folder, image_filename)
            shutil.copyfile(image_path, original_output_path)
