import os
import shutil
import random
from math import floor

path = "."


# create file list for all images
def get_file_list_from_dir(datadir):
    all_files = os.listdir(os.path.join(path, datadir))
    data_files = list(filter(lambda file: file.endswith('.jpg'), all_files))
    return data_files

    
# split train and test files
def get_training_and_testing_files(file_list):
    split = 0.90
    split_index = floor(len(file_list) * split)
    training = file_list[:split_index]
    testing = file_list[split_index:]
    return training, testing


# split train and val files
def get_training_and_validation_files(file_list):
    split = 0.83
    split_index = floor(len(file_list) * split)
    training = file_list[:split_index]
    validation = file_list[split_index:]
    return training, validation


folders = ["0_Negative", "1_Primordial", "2_Primary", "3_Secondary", "4_Tertiary"]


for fd in folders:
    os.makedirs(os.path.join(path, "train", fd), exist_ok=True)
    os.makedirs(os.path.join(path, "test", fd), exist_ok=True)
    file_list_all = get_file_list_from_dir(os.path.join(path, "images", fd))
    # shuffle file list
    random.Random(42).shuffle(file_list_all)
    train_list, test_list = get_training_and_testing_files(file_list_all)
    # copy train and test files in directories
    for f in train_list:
        shutil.copy(os.path.join(path, "images", fd, f), os.path.join(path, "train", fd))
    for f in test_list:
        shutil.copy(os.path.join(path, "images", fd, f), os.path.join(path, "test", fd))


for fd in folders:
    os.makedirs(os.path.join(path, "val", fd), exist_ok=True)
    file_list_train = get_file_list_from_dir(os.path.join(path, "train", fd))
    random.Random(42).shuffle(file_list_train)
    train_list, val_list = get_training_and_validation_files(file_list_train)
    for f in val_list:
        shutil.move(os.path.join(path, "train", fd, f), os.path.join(path, "val", fd))