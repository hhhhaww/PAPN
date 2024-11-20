import torch.distributed as dist
import torch.multiprocessing as mp
import torch
from torch.nn import SyncBatchNorm
from torch import nn
import timm
import os
import argparse
from torch.optim import lr_scheduler
import time
from utils.tools import AverageMeter, ProgressMeter, get_logger
import random
import numpy as np
from pathlib import Path
from torch import optim
from datetime import datetime
from torch.nn import functional as F
from utils.tools import read_config
from utils import augments
from torchvision.datasets import ImageFolder
from utils.tools import concat_all_gather


def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


class Encoder(nn.Module):
    def __init__(self, base_encoder, num_classes):
        super().__init__()

        self.encoder = timm.create_model(
            base_encoder,
            pretrained=False,
            num_classes=0
        )

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(2048 * 2, num_classes)

        self.part_proto = nn.Parameter(torch.zeros(5, 2048))

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def get_feat_part(self, feat, part_proto):
        N, _, H, W = feat.shape
        M = part_proto.size(0)

        feat_flat = feat.flatten(2).permute((0, 2, 1))
        feat_norm = F.normalize(feat_flat, dim=-1)
        part_proto_norm = F.normalize(part_proto, dim=-1)
        feat_sim = (feat_norm @ part_proto_norm.T).permute((0, 2, 1))

        feat_sim = feat_sim.reshape((N, M, H, W))
        feat_parts = feat_sim.unsqueeze(2) * feat.unsqueeze(1)
        feat_parts = feat_parts.flatten(3).sum(-1)

        feat_part = feat_parts.max(dim=1)[0]

        return feat_part

    def forward(self, im):
        feat = self.encoder.forward_features(im)
        feat_global = self.pool(feat)
        feat_part = self.get_feat_part(feat, self.part_proto)
        feat_global_part = torch.concatenate([feat_global, feat_part], dim=1)

        feat_proj = self.fc(feat_global_part)

        return feat_global_part, feat_proj


def load_resnet(gpu, cfg):
    if cfg.test.model_path != None:
        checkpoint = torch.load(cfg.test.model_path, map_location='cpu')
    else:
        path = os.path.join(cfg.base.log_path, 'best-train.pt')
        checkpoint = torch.load(path, map_location='cpu')

    assert isinstance(checkpoint, dict), type(checkpoint)

    if 'model' in checkpoint.keys():
        checkpoint = checkpoint['model']

    for k in list(checkpoint.keys()):
        if k.startswith("encoder_q.encoder"):
            checkpoint[k[len("encoder_q."):]] = checkpoint[k]
        elif k.startswith("part_proto"):
            continue
        del checkpoint[k]

    model = Encoder(cfg.model.params.base_encoder, cfg.data.num_class).cuda(gpu)
    model.load_state_dict(checkpoint, strict=False)

    return model


def run(gpu, args):
    cfg = read_config(args.config_file)

    same_seeds(cfg.base.seed)

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=gpu
    )

    torch.cuda.set_device(gpu)
    torch.cuda.empty_cache()

    save_path = Path(cfg.base.log_path)
    save_path.mkdir(exist_ok=True, parents=True)
    log_path = save_path / 'lincls.log'
    logger = get_logger(log_path)

    if dist.get_rank() == 0:
        now = datetime.now()
        logger.info(f'\n=================== Start at {now.hour}:{now.minute}:{now.second} ===================')
        logger.info(cfg)
        logger.info('\n')

    model = load_resnet(gpu, cfg)

    batch_size = cfg.test.batch_size // dist.get_world_size()
    if dist.is_initialized() and dist.get_world_size() > 1:
        model = SyncBatchNorm.convert_sync_batchnorm(model)

    for name, param in model.named_parameters():
        if name not in ["fc.weight", "fc.bias"]:
            param.requires_grad = False

    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    optimizer = optim.SGD(
        ddp_model.module.fc.parameters(),
        lr=cfg.test.lr,
        momentum=0.9
    )

    criterion = nn.CrossEntropyLoss().cuda(gpu)

    data_path = Path(cfg.data.data_path)

    train_dataset = ImageFolder(
        str(data_path / 'train'),
        transform=getattr(augments, cfg.test.train_aug)()
    )

    val_dataset = ImageFolder(
        str(data_path / 'test'),
        transform=getattr(augments, cfg.test.val_aug)()
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=gpu
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=args.world_size,
        rank=gpu
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers // dist.get_world_size(),
        pin_memory=True,
        sampler=train_sampler
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers // dist.get_world_size(),
        pin_memory=True,
        sampler=val_sampler
    )

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=cfg.test.epochs)

    logger.info(f'gpu {dist.get_rank()} begins to train')

    best_acc1 = 0
    for epoch in range(cfg.test.epochs):

        begin_time = time.time()

        train_loader.sampler.set_epoch(epoch)

        train(ddp_model, optimizer, criterion, train_loader, epoch, gpu, logger)

        rank1, rank5, acc1, acc5 = validate(ddp_model, val_loader, gpu)

        scheduler.step()

        end_time = time.time()
        if dist.get_rank() == 0:

            if acc1 > best_acc1:
                best_acc1 = acc1
                checkpoint = {
                    'model': ddp_model.module.state_dict(),
                    'rank1': rank1,
                    'rank5': rank5,
                    'acc1': acc1,
                    'acc5': acc5
                }
                torch.save(checkpoint, str(save_path / f'best-lincls.pt'))

            logger.info('    '.join([f'rank1={rank1:.2f}',
                                     f'rank5={rank5:.2f}',
                                     f'acc1={acc1:.2f}',
                                     f'acc5={acc5:.2f}',
                                     f'time_cost={(end_time - begin_time):.2f}s',
                                     f'best_acc1={best_acc1:.2f}']))

    if dist.get_rank() == 0:
        now = datetime.now()
        logger.info(f'==================== End at {now.hour}:{now.minute}:{now.second} ====================\n')


def train(ddp_model, optimizer, criterion, train_loader, epoch, gpu, logger):
    ddp_model.eval()

    BATCH_TIME = AverageMeter("Time", ":6.3f")
    DATA_TIME = AverageMeter("Data", ":6.3f")
    LOSS = AverageMeter("Loss", ":.4e")
    LR = AverageMeter("lr", ":.4e")
    TOP1 = AverageMeter("Acc@1", ":6.2f")
    TOP5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [BATCH_TIME, DATA_TIME, LOSS, LR, TOP1, TOP5],
        prefix="Epoch: [{}]".format(epoch),
        logger=logger
    )

    end = time.time()
    for i, (images, labels) in enumerate(train_loader):
        DATA_TIME.update(time.time() - end)
        LR.update(optimizer.param_groups[0]["lr"])

        images = images.cuda(gpu, non_blocking=True)
        labels = labels.cuda(gpu, non_blocking=True)

        _, output = ddp_model(images)

        loss = criterion(output, labels)

        acc1, acc5 = accuracy(output, labels, topk=(1, 5))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dist.barrier()

        gathered_info = torch.tensor([loss, acc1, acc5]).cuda(gpu)
        dist.all_reduce(gathered_info, op=torch.distributed.ReduceOp.AVG)
        gathered_info = gathered_info.tolist()
        LOSS.update(gathered_info[0])
        TOP1.update(gathered_info[1])
        TOP5.update(gathered_info[2])
        BATCH_TIME.update(time.time() - end)
        end = time.time()

    if dist.get_rank() == 0:
        progress.display(len(train_loader))


@torch.no_grad()
def validate(ddp_model, val_loader, gpu):
    ddp_model.eval()

    ps = []

    rls = []

    for batch, (images, labels) in enumerate(val_loader):
        images = images.cuda(gpu, non_blocking=True)
        labels = labels.cuda(gpu, non_blocking=True).reshape(-1, 1)
        repre, pred = ddp_model(images)
        rl = torch.concatenate([repre, labels], dim=1)
        ps.append(pred)
        rls.append(rl)

    rls_cat = torch.concatenate(rls, dim=0)
    ps_cat = torch.concatenate(ps, dim=0)

    all_rls = concat_all_gather(rls_cat)
    all_ps = concat_all_gather(ps_cat)

    all_pred = all_ps
    all_repre = all_rls[:, :-1]
    all_label = all_rls[:, -1]

    acc1, acc5 = accuracy(all_pred, all_label.flatten(), topk=(1, 5))

    rank1, rank5 = cal_rank(all_repre, all_label)

    return rank1, rank5, acc1, acc5


def cal_rank(repre, label):
    N = repre.size(0)

    all_label_repeat = label.repeat(N, 1)

    all_repre_l2 = F.normalize(repre, dim=-1, p=2)
    similarity = all_repre_l2 @ all_repre_l2.T
    mask = torch.eye(similarity.size(0)).bool()
    similarity[mask] = 0

    _, top5_indices = torch.topk(similarity, k=5, dim=-1, largest=True, sorted=True)
    top5_labels = all_label_repeat.gather(1, top5_indices)
    top1_labels = top5_labels[:, 0]

    rank1 = torch.eq(top1_labels, label).float().sum() / N * 100
    rank5 = torch.any(top5_labels == label[:, None], dim=1).float().sum() / N * 100

    return rank1, rank5


if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    gpus = torch.cuda.device_count()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str, required=True)

    args = parser.parse_args()
    args.world_size = gpus

    mp.spawn(run, nprocs=gpus, args=(args,))
