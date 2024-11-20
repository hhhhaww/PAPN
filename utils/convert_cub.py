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
    if not os.path.exists('./dataset/cub/data'):
        os.mkdir('./dataset/cub/data')
    zip_ref.extractall('./dataset/cub/data')

id2path = {}
with open(f'./dataset/cub/data/CUB_200_2011/images.txt') as f:
    for line in f:
        image_id, path = line.split()
        id2path[image_id] = path

id2class = {}
with open(f'./dataset/cub/data/CUB_200_2011/image_class_labels.txt') as f:
    for line in f:
        image_id, class_id = line.split()
        id2class[image_id] = int(class_id) - 1

train_id = []
test_id = []
with open(f'./dataset/cub/data/CUB_200_2011/train_test_split.txt') as f:
    for line in f:
        image_id, is_train = line.split()
        if bool(int(is_train)):
            train_id.append(image_id)
        else:
            test_id.append(image_id)

train_path = Path('./dataset/cub/train')
test_path = Path('./dataset/cub/test')

for each_id in train_id:
    img_name = id2path[each_id].split('/')[1]
    img_path = Path(f'./dataset/cub/data/CUB_200_2011/images/{id2path[each_id]}')
    if not os.path.exists(f'./dataset/cub/train/{id2class[each_id]}'):
        os.makedirs(f'./dataset/cub/train/{id2class[each_id]}')
    img_dest_path = Path(f'./dataset/cub/train/{id2class[each_id]}/{img_name}')
    shutil.copy(img_path, img_dest_path)

for each_id in test_id:
    img_name = id2path[each_id].split('/')[1]
    img_path = Path(f'./dataset/cub/data/CUB_200_2011/images/{id2path[each_id]}')
    if not os.path.exists(f'./dataset/cub/test/{id2class[each_id]}'):
        os.makedirs(f'./dataset/cub/test/{id2class[each_id]}')
    img_dest_path = Path(f'./dataset/cub/test/{id2class[each_id]}/{img_name}')
    shutil.copy(img_path, img_dest_path)
