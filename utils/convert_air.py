import argparse
from pathlib import Path
import os
import shutil
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str, required=True)
args = parser.parse_args()

dataset_path = args.path
with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
    if not os.path.exists('./dataset/air'):
        os.mkdir('./dataset/air')
    zip_ref.extractall('./dataset/air/data')

if not os.path.exists('./dataset/air/train'):
    os.mkdir('./dataset/air/train')
if not os.path.exists('./dataset/air/test'):
    os.mkdir('./dataset/air/test')

root = Path('./dataset/air/data/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data')

with open(root / 'images_variant_trainval.txt', "r") as f:
    train_data = [line.strip().split(" ", maxsplit=1) for line in f]

with open(root / 'images_variant_test.txt', "r") as f:
    val_data = [line.strip().split(" ", maxsplit=1) for line in f]

for i, data in enumerate(train_data):
    print(f'\r{i}/{len(train_data)}', end='')
    path = root / 'images' / f'{data[0]}.jpg'
    if '/' in data[1]:
        data[1] = data[1].replace('/', '')
    label = os.path.join('./dataset/air/train', data[1])
    if not os.path.exists(label):
        os.mkdir(label)
    shutil.copy(path, label)
print()

for i, data in enumerate(val_data):
    print(f'\r{i}/{len(val_data)}', end='')
    path = root / 'images' / f'{data[0]}.jpg'
    if '/' in data[1]:
        data[1] = data[1].replace('/', '')
    label = os.path.join('./dataset/air/test', data[1])
    if not os.path.exists(label):
        os.mkdir(label)
    shutil.copy(path, label)