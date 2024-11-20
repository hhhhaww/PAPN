import torch
import torch.nn as nn
import timm
from torch.nn import functional as F
import torch.distributed as dist

layer_dims = {
    'layer1': 256,
    'layer2': 512,
    'layer3': 1024,
    'layer4': 2048
}


class AlignLoss(nn.Module):
    def __init__(self, t_q=1, t_k=1):
        super().__init__()

        self.t_q = t_q
        self.t_k = t_k

        self.loss_fn = nn.MSELoss()

    def self_dist(self, q, k):
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        loss = self.loss_fn(q, k)

        return loss

    def forward(self, q_feats, k_feats):
        loss = torch.tensor(0).to(q_feats[0])
        for q_feat, k_feat in zip(q_feats, k_feats):
            loss += self.self_dist(q_feat, k_feat)

        return loss


class Encoder(nn.Module):
    def __init__(self,
                 base_encoder,
                 proj_dim,
                 layer_proj_dim,
                 layer_names,
                 pretrain):

        super().__init__()

        self.num_layers = len(layer_names)

        self.encoder = timm.create_model(
            base_encoder, pretrained=False, num_classes=0
        )

        if pretrain is not None:
            checkpoint = torch.load(pretrain, map_location="cpu")
            for key in list(checkpoint.keys()):
                if key.__contains__("fc"):
                    del checkpoint[key]
            self.encoder.load_state_dict(checkpoint, strict=True)

        self.layers_fc = nn.ModuleList([])
        self.layers_emb = []

        dims = [layer_dims[layer] for layer in layer_names]
        part_dim = dims[-1]

        for layer_name, dim in zip(layer_names, dims):
            layer = self._find_layer(layer_name)
            layer.register_forward_hook(self._hook)
            self.layers_fc.append(nn.Linear(dim, layer_proj_dim))

        self.fc = nn.Sequential(
            nn.Linear(sum(dims) + part_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

    def _hook(self, _, __, output):
        self.layers_emb.append(output)

    def _clear_layers_emb(self):
        self.layers_emb = []

    def _find_layer(self, layer_name):
        if type(layer_name) == str:
            modules = dict([*self.encoder.named_modules()])
            return modules.get(layer_name, None)
        elif type(layer_name) == int:
            children = [*self.encoder.children()]
            return children[layer_name]
        return None

    def _get_part_feature(self, feat, part_proto):
        N, _, H, W = feat.shape
        M = part_proto.size(0)
        feat_flat = feat.flatten(2).permute((0, 2, 1))
        feat_norm = F.normalize(feat_flat, dim=-1)
        part_proto_norm = F.normalize(part_proto, dim=-1)
        feat_sim = (feat_norm @ part_proto_norm.T).permute((0, 2, 1))
        feat_sim = feat_sim.reshape((N, M, H, W))
        feat_parts = feat_sim.unsqueeze(2) * feat.unsqueeze(1)
        feat_parts = feat_parts.flatten(3).sum(-1)
        feat_part = feat_parts.mean(dim=1)
        return feat_part

    def forward(self, im, part_proto):

        if len(im.shape) == 5:
            feats_proj = []
            for i in range(self.num_layers):
                _ = self.encoder(im[:, i])
                feat_layer = self.layers_emb[i]
                feat_layer_pool = self.pool(feat_layer)
                feat_layer_proj = self.layers_fc[i](feat_layer_pool)
                feats_proj.append(feat_layer_proj)
                self._clear_layers_emb()

            _ = self.encoder(im[:, self.num_layers])
            feat_layers_pool = [self.pool(each) for each in self.layers_emb]
            feat_global = torch.concatenate(feat_layers_pool, dim=1)
            feat_part = self._get_part_feature(self.layers_emb[-1], part_proto)
            feat_global_part = torch.concatenate([feat_global, feat_part], dim=1)
            feat_global_part_proj = self.fc(feat_global_part)
            self._clear_layers_emb()
            return feats_proj, feat_global_part_proj
        else:
            _ = self.encoder(im)
            feat = self.layers_emb[-1]
            feat_global = self.pool(feat)
            feat_part = self._get_part_feature(feat, part_proto)
            feat_global_part = torch.concatenate([feat_global, feat_part], dim=1)
            return feat_global_part, feat_global


class PAPN(nn.Module):

    def __init__(
            self,
            base_encoder,
            proj_dim=256,
            layer_proj_dim=128,
            layer_names=None,
            K=4096,
            m=0.999,
            n_parts=5,
            T=0.15,
            pretrain=None):

        super(PAPN, self).__init__()

        if layer_names is None:
            layer_names = ['layer2', 'layer3', 'layer4']

        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = Encoder(base_encoder, proj_dim, layer_proj_dim, layer_names, pretrain)
        self.encoder_k = Encoder(base_encoder, proj_dim, layer_proj_dim, layer_names, pretrain)

        for param_q, param_k in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", F.normalize(torch.randn(proj_dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.ssl_loss = nn.CrossEntropyLoss()
        self.sd_loss = AlignLoss()

        self.part_proto = nn.Parameter(
            generate_orthonormal_vectors(
                n_parts, layer_dims[layer_names[-1]]
            )
        )

    @torch.no_grad()
    def _momentum_update_key_encoder(self):

        for param_q, param_k in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):

        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        idx_shuffle = torch.randperm(batch_size_all).cuda()

        torch.distributed.broadcast(idx_shuffle, src=0)

        idx_unshuffle = torch.argsort(idx_shuffle)

        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):

        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):

        q_feats, q = self.encoder_q(im_q, self.part_proto)
        q_norm = F.normalize(q, dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k_feats, k = self.encoder_k(im_k, self.part_proto)
            k_norm = F.normalize(k, dim=1)

            for i in range(len(k_feats)):
                k_feats[i] = self._batch_unshuffle_ddp(k_feats[i], idx_unshuffle)

            k_norm = self._batch_unshuffle_ddp(k_norm, idx_unshuffle)

        sd_loss = self.sd_loss(q_feats, k_feats)

        l_pos = torch.einsum("nc,nc->n", [q_norm, k_norm]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q_norm, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        self._dequeue_and_enqueue(k_norm)

        ssl_loss = self.ssl_loss(logits, labels)

        return ssl_loss + sd_loss


@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def generate_orthonormal_vectors(n, dim):
    A = torch.randn(dim, n)
    U, S, Vt = torch.svd(A)
    return U.T