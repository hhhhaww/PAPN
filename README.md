# PAPN

## Abstract

In this paper, we mitigate the problem of Self-Supervised Learning (SSL) for fine-grained representation learning, aimed at distinguishing subtle differences within highly similar subordinate categories. Our preliminary analysis shows that SSL, especially the multi-stage alignment strategy, performs well on generic categories but struggles with fine-grained distinctions. To overcome this limitation, we propose a prototype-based contrastive learning module with stage-wise progressive augmentation. Unlike previous methods, our stage-wise progressive augmentation adapts data augmentation across stages to better suit SSL on fine-grained datasets. The prototype-based contrastive learning module captures both holistic and partial patterns, extracting global and local image representations to enhance feature discriminability. Experiments on popular fine-grained benchmarks for classification and retrieval tasks demonstrate the effectiveness of our method, and extensive ablation studies confirm the superiority of our proposals.

## Dataset

You can download the dataset from the following websites:

| Dataset       | URL                                                          |
| ------------- | ------------------------------------------------------------ |
| FGVC Aircraft | https://www.kaggle.com/datasets/seryouxblaster764/fgvc-aircraft |
| CUB200-2011   | https://www.kaggle.com/datasets/wenewone/cub2002011          |
| Stanford Cars | https://www.kaggle.com/datasets/rickyyyyyyy/torchvision-stanford-cars |

Then you should run the following commands:

```bash
cd PAPN
python ./utils/convert_cub.py -p path_of_cub.zip
python ./utils/convert_car.py -p path_of_car.zip
python ./utils/convert_air.py -p path_of_air.zi
```

The final datasets structure are as follows:

```
PAPN
|-- dataset
    |-- cub
        |-- train/
        |-- test/
    |-- car
        |-- train/
        |-- test/
    |-- air
        |-- train/
        |-- test/
```

## Run the code

- The experiments are carried out on Ubuntu 18.04.6 LTS, utilizing four GeForce RTX 3090 GPUs, each with 24GB of memory. The CUDA version employed is 11.6.
- Install the required packages:

```bash
pip install -r requirements.txt
```

- The running commands for training:

```bash
python main_train.py -c configs/papn_air.yaml
python main_train.py -c configs/papn_car.yaml
python main_train.py -c configs/papn_cub.yaml
```

- The running commands for linear probing and retrieval:

```
python main_lincls.py -c configs/papn_air.yaml
python main_lincls.py -c configs/papn_car.yaml
python main_lincls.py -c configs/papn_cub.yaml
```