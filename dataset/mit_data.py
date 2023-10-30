import os
import shutil

# Create a folder named 'test' if it doesn't exist
target_folder = '/home/hung/mit_indoor/test'
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

src = '/home/hung/mit_indoor/indoorCVPR_09/Images'
# Read the file and move images to the 'test' folder
with open('/home/hung/mit_indoor/TestImages.txt', 'r') as file:
    for line in file:
        # Remove newline characters from the line
        line = line.strip()

        # Extract the folder name and image file name
        folder_name, image_name = line.split('/')

        # Create the destination folder path
        destination_folder = os.path.join(target_folder, folder_name)

        # Create the destination folder if it doesn't exist
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Construct the source and destination file paths
        source_path = os.path.join(src, folder_name, image_name)
        destination_path = os.path.join(destination_folder, image_name)

        # Move the image to the 'test' folder
        shutil.move(source_path, destination_path)
        print(destination_path)

print("Images moved successfully to the 'test' folder.")
