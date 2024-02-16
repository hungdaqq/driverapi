import os
import shutil
import pandas as pd

# Read the CSV file
df = pd.read_csv('/home/hung/auc.distracted.driver.dataset_v2/v1/Train_data_list.csv')

# Iterate through rows and move images to corresponding label folders
for index, row in df.iterrows():
    image_path = row['Image']
    label = row['Label']

    path = image_path.split('/')

    image_path = '/home/hung/auc.distracted.driver.dataset_v2/v1/' + path[2] + '/' +  path[3]

    print(image_path)

    # Create label folder if not exists
    label_folder = 'train' + f'/{label}/'
    os.makedirs(label_folder, exist_ok=True)
    
    # Move the image to the label folder
    shutil.move(image_path, label_folder + os.path.basename(image_path))

print('Images moved successfully.')
