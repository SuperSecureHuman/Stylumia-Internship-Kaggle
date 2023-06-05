import os
import shutil
import csv

def create_folders_and_move_files(csv_file, input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the CSV file
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row if present

        for row in csv_reader:
            filename, label = row
            label_folder = os.path.join(output_folder, label)

            # Create the label folder if it doesn't exist
            os.makedirs(label_folder, exist_ok=True)

            # Move the file to the respective label folder
            source_path = os.path.join(input_folder, filename)
            destination_path = os.path.join(label_folder, filename)
            shutil.move(source_path, destination_path)

            print(f"Moved {filename} to {label_folder}")

# Specify the CSV file path, input folder path, and the output folder path
csv_file_path = '/home/venom/repo/Stylumia-Internship-Kaggle/Dataset/train.csv'
input_folder_path = '/home/venom/repo/Stylumia-Internship-Kaggle/Dataset/train'
output_folder_path = '/home/venom/repo/Stylumia-Internship-Kaggle/Dataset/Modified'

# Call the function to create folders and move files
create_folders_and_move_files(csv_file_path, input_folder_path, output_folder_path)
