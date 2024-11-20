import os
import logging
import yaml
import torch

class YamlConfig:
    def __init__(self, yaml_dict):
        for key, value in yaml_dict.items():
            if isinstance(value, dict):
                setattr(self, key, YamlConfig(value))
            else:
                setattr(self, key, value)
        self.yaml_string = yaml.dump(yaml_dict, default_flow_style=False)

    def __repr__(self):
        return self.yaml_string

    def __getattribute__(self, name):
        if name == '__dict__':
            return {k: v for k, v in super().__getattribute__(name).items() if k != 'yaml_string'}
        return super().__getattribute__(name)


class AverageMeter:
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, logger=None, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if self.logger:
            self.logger.info("\t".join(entries))
        else:
            print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def get_logger(log_path):
    if not os.path.exists(os.path.dirname(log_path)):
        os.mkdir(os.path.dirname(log_path))

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(filename=log_path, mode='a')
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def read_config(file_path):
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    config = YamlConfig(config_dict)
    return config
