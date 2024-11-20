import argparse
import os
import shutil
import scipy
import zipfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str, required=True)
args = parser.parse_args()

dataset_path = args.path
with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
    if not os.path.exists('./dataset/car/data'):
        os.mkdir('./dataset/car/data')
    zip_ref.extractall('./dataset/car/data')

if not os.path.exists('./dataset/car/train'):
    os.makedirs('./dataset/car/train')
if not os.path.exists('./dataset/car/test'):
    os.makedirs('./dataset/car/test')

root = Path('./dataset/car/data/stanford_cars')

train_path = root / 'cars_train'
train_annos = scipy.io.loadmat(str(root / 'devkit/cars_train_annos.mat'))
train_annos = train_annos['annotations'].flatten()
train_data_label = [(train_path / anno[5].item(), anno[4].item()) for anno in train_annos]
for i, (image_path, label) in enumerate(train_data_label):
    print(f'\r{i}/{len(train_data_label)}', end='')
    new_path = os.path.join('./dataset/car/train', str(label))
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    shutil.copy(image_path, new_path)
print()

test_path = root / 'cars_test'
test_annos = scipy.io.loadmat(str(root / 'cars_test_annos_withlabels.mat'))
test_annos = test_annos['annotations'].flatten()
test_data_label = [(test_path / anno[5].item(), anno[4].item()) for anno in test_annos]
for i, (image_path, label) in enumerate(test_data_label):
    print(f'\r{i}/{len(test_data_label)}', end='')
    new_path = os.path.join('./dataset/car/test', str(label))
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    shutil.copy(image_path, new_path)
print()
