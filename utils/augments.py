from torchvision import transforms
import random
from PIL import ImageFilter
import torch


class GaussianBlur:
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

    def __repr__(self):
        param_str = ', '.join(f'{key}={getattr(self, key)}' for key in dir(self) if not key.startswith('__'))
        string = f"{self.__class__.__name__}({param_str})"
        return string


class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = []
        if type(base_transform) == list:
            assert len(base_transform) == 2
            self.base_transform = base_transform
        else:
            self.base_transform.append(base_transform)
            self.base_transform.append(base_transform)

    def __call__(self, x):
        q = self.base_transform[0](x)
        k = self.base_transform[1](x)
        return [q, k]

    def __repr__(self):
        return repr(self.base_transform)


class MultiCropsTransform:
    def __init__(self, transforms):
        assert type(transforms) == list

        self.q_trans = transforms[0]
        self.k_trans = transforms[1]
        self.base_trans = transforms[2]

    def __call__(self, x):
        q = []
        k = []

        for tran in self.q_trans:
            q.append(tran(x))
        q.append(self.base_trans(x))

        for tran in self.k_trans:
            k.append(tran(x))
        k.append(self.base_trans(x))

        q = torch.stack(q, dim=0)
        k = torch.stack(k, dim=0)

        return [q, k]

    def __repr__(self):
        return repr(self.base_trans)


class MultiLayerTransform:

    def __init__(self, q_layer_trans, k_layer_trans, trans):
        self.q_layer_trans = q_layer_trans
        self.k_layer_trans = k_layer_trans
        self.trans = trans

    def __call__(self, x):
        q = []
        k = []

        for tran in self.q_layer_trans:
            q.append(tran(x))
        q.append(self.trans(x))

        for tran in self.k_layer_trans:
            k.append(tran(x))
        k.append(self.trans(x))

        q = torch.stack(q, dim=0)
        k = torch.stack(k, dim=0)

        return [q, k]


def linear_aug(s, q_scale, k_scale):
    def multi_layer_aug(s, scale):
        aug = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(scale[1] - (scale[1] - scale[0]) * s, scale[1])),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)],
                p=0.8 * s),
            transforms.RandomGrayscale(p=0.2 * s),
            transforms.RandomApply(
                [GaussianBlur([0.1 * s, 2.0 * s])], p=0.5 * s),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        return aug

    q_layer_trans = [multi_layer_aug(s_each, q_scale) for s_each in s]
    k_layer_trans = [multi_layer_aug(s_each, k_scale) for s_each in s]
    trans = multi_layer_aug(1, [0.2, 1])

    return MultiLayerTransform(q_layer_trans, k_layer_trans, trans)


def mocov2_aug():
    return TwoCropsTransform(transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]))


def byol_aug():
    return TwoCropsTransform([
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.1),
            transforms.RandomSolarize(threshold=0.5, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    ])


def train_aug():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def val_aug():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
