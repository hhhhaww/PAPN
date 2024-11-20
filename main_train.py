import argparse
import random
import time
from pathlib import Path
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import SyncBatchNorm
from torch.optim import lr_scheduler
import models
from utils.tools import AverageMeter, ProgressMeter, get_logger
from torch import optim
from datetime import datetime
from torch.nn import functional as F
from utils.tools import concat_all_gather
from utils.tools import read_config
import os
from utils import augments
from torchvision.datasets import ImageFolder


def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
    log_path = save_path / f'train.log'
    logger = get_logger(log_path)

    if dist.get_rank() == 0:
        now = datetime.now()
        logger.info(f'\n=================== Start at {now.hour}:{now.minute}:{now.second} ===================')
        logger.info(cfg)
        logger.info('\n')

    model = getattr(models, cfg.model.name)(**cfg.model.params.__dict__).cuda(gpu)

    batch_size = cfg.train.batch_size // dist.get_world_size()
    if dist.is_initialized() and dist.get_world_size() > 1:
        model = SyncBatchNorm.convert_sync_batchnorm(model)

    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=False)

    optimizer = optim.SGD(ddp_model.parameters(), lr=cfg.train.lr, momentum=0.9, weight_decay=1e-4)

    train_transform = augments.linear_aug(s=cfg.train.s, q_scale=cfg.train.q_scale, k_scale=cfg.train.k_scale)

    val_transform = getattr(augments, cfg.train.val_aug)()

    scaler = GradScaler()

    data_path = Path(cfg.data.data_path)

    train_dataset = ImageFolder(str(data_path / 'train'), transform=train_transform)

    val_dataset = ImageFolder(str(data_path / 'test'), transform=val_transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=gpu
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers // dist.get_world_size(),
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=args.world_size,
        rank=gpu
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers // dist.get_world_size(),
        pin_memory=True,
        drop_last=True,
        sampler=val_sampler
    )

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)

    start_epoch = 0

    if cfg.base.resume_path is not None:
        checkpoint = torch.load(cfg.base.resume_path, map_location='cpu')
        ddp_model.module.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['start_epoch']
        logger.info(f'gpu {dist.get_rank()} loads checkpoint {cfg.base.resume_path}')

    ddp_model.train()

    logger.info(f'gpu {dist.get_rank()} begins to train')

    best_rank5 = 0

    best_rank1 = 0

    for epoch in range(start_epoch, cfg.train.epochs):

        begin_time = time.time()

        train_loader.sampler.set_epoch(epoch)

        train(ddp_model, optimizer, train_loader, epoch, gpu, scaler, logger, cfg)

        rank1, rank5 = validate(ddp_model, val_loader, gpu)

        scheduler.step()


        if rank5 > best_rank5 and dist.get_rank() == 0:
            best_rank5 = rank5
            checkpoint = {
                'model': ddp_model.module.state_dict(),
                'rank1': rank1,
                'rank5': rank5
            }
            checkpoint_path = save_path / f'best-train.pt'
            torch.save(checkpoint, str(checkpoint_path))

        if rank1 > best_rank1 and dist.get_rank() == 0:
            best_rank1 = rank1
            checkpoint = {
                'model': ddp_model.module.state_dict(),
                'rank1': rank1,
                'rank5': rank5
            }
            checkpoint_path = save_path / f'best-train-rank1.pt'
            torch.save(checkpoint, str(checkpoint_path))

        if (epoch + 1) % cfg.train.save_every_epoch == 0 and dist.get_rank() == 0:
            checkpoint = {
                'model': ddp_model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'start_epoch': epoch + 1,
                'rank1': rank1,
                'rank5': rank5
            }
            checkpoint_path = save_path / f'checkpoint.{epoch}.pt'
            torch.save(checkpoint, str(checkpoint_path))

        end_time = time.time()

        if dist.get_rank() == 0:
            logger.info('    '.join([f'rank1={rank1:.2f}',
                                     f'rank5={rank5:.2f}',
                                     f'best_rank5={best_rank5:.2f}',
                                     f'time_cost={(end_time - begin_time):.2f}s', ]))

    if dist.get_rank() == 0:
        now = datetime.now()
        logger.info(f'==================== End at {now.hour}:{now.minute}:{now.second} ====================\n')


def train(ddp_model, optimizer, train_loader, epoch, gpu, scaler, logger, cfg):
    ddp_model.train()

    BATCH_TIME = AverageMeter("Time", ":6.3f")
    DATA_TIME = AverageMeter("Data", ":6.3f")
    LOSS = AverageMeter("loss", ":.4e")
    LR = AverageMeter("lr", ":.4e")

    progress_meter = ProgressMeter(
        len(train_loader),
        [BATCH_TIME, DATA_TIME, LOSS, LR],
        prefix="Epoch: [{}]".format(epoch),
        logger=logger
    )

    end = time.time()
    for batch, (images, idx) in enumerate(train_loader):

        DATA_TIME.update(time.time() - end)
        LR.update(optimizer.param_groups[0]["lr"])

        image_one = images[0].cuda(gpu, non_blocking=True)
        image_two = images[1].cuda(gpu, non_blocking=True)

        optimizer.zero_grad()

        if cfg.train.amp:
            with autocast():
                loss = ddp_model(image_one, image_two)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = ddp_model(image_one, image_two)
            loss.backward()
            optimizer.step()

        dist.barrier()

        dist.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
        LOSS.update(loss.item())

        BATCH_TIME.update(time.time() - end)
        end = time.time()

        if dist.get_rank() == 0:
            if (batch + 1) % cfg.train.check_every_batch == 0 or (batch + 1) % len(train_loader) == 0:
                progress_meter.display((batch + 1))


@torch.no_grad()
def validate(ddp_model, val_loader, gpu):
    ddp_model.eval()

    rls = []

    for _, (images, labels) in enumerate(val_loader):
        images = images.cuda(gpu, non_blocking=True)
        labels = labels.cuda(gpu, non_blocking=True).reshape(-1, 1)
        repre, g_repre = ddp_model.module.encoder_q(images, ddp_model.module.part_proto)
        rl = torch.concatenate([repre, labels], dim=1)
        rls.append(rl)

    rls_cat = torch.concatenate(rls, dim=0)

    all_rls = concat_all_gather(rls_cat)

    all_repre = all_rls[:, :-1]
    all_label = all_rls[:, -1]

    rank1, rank5 = cal_rank(all_repre, all_label)

    return rank1, rank5

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
