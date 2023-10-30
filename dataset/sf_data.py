import os
import shutil
from sklearn.model_selection import train_test_split

# Replace this with the path to your images folder
source_folder = '/home/hung/statefarm/imgs/train/'

# Replace this with the path to your destination folder for training images
train_destination = '/home/hung/statefarm/train/'

# Replace this with the path to your destination folder for testing images
test_destination = '/home/hung/statefarm/test/'

# Function to copy files from source to destination
def copy_files(file_list, source, destination):
    for file in file_list:
        source_path = os.path.join(source, file)
        destination_path = os.path.join(destination, file)
        shutil.copyfile(source_path, destination_path)

# Split the dataset
for cls in os.listdir(source_folder):
    cls_dir = os.path.join(source_folder,cls)
    cls_files = os.listdir(cls_dir)
    train_images, test_images = train_test_split(cls_files, test_size=0.2, random_state=42)
    new_train_dst = os.path.join(train_destination,cls)
    new_test_dst = os.path.join(test_destination,cls)
    # Create directories if they don't exist
    os.makedirs(new_train_dst, exist_ok=True)
    os.makedirs(new_test_dst, exist_ok=True)
    copy_files(train_images, cls_dir, new_train_dst)
    copy_files(test_images, cls_dir, new_test_dst)
    print(cls)