import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from protonets.models import register_model

from .utils import euclidean_dist

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Protonet(nn.Module):
    def __init__(self, encoder):
        super(Protonet, self).__init__()
        
        self.encoder = encoder

    def loss(self, sample):
        xs = Variable(sample['xs']) # support  CLASS N channel w h
        xq = Variable(sample['xq']) # query    CLASS N channel w h

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()
        # shape [600,1,28,28] support set query set 合并
        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)
        # shape [600,64]
        z = self.encoder.forward(x)
        z_dim = z.size(-1)
        # latent = z[:n_class*n_support].view(n_class, n_support, z_dim)
        # result = latent.mean(1)
        # 计算support同一种类别的多个sample的均值
        z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1) # shape [60,64]
        zq = z[n_class*n_support:] # shape [300,64]
        # 每一类别的样本量与每一类别的距离
        dists = euclidean_dist(zq, z_proto) # 欧几里得距离
        # logsoft = F.log_softmax(-dists, dim=1)
        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1) #shape [60 , 5 , 60]
        # gathery = -log_p_y.gather(2, target_inds) 每个样本的logpy值
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2) #y_hat [60 , 5]
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }

    def mixuploss(self,sample,ya,yb,lam):

        xs = Variable(sample['xs']) # support  CLASS N channel w h
        xq = Variable(sample['xq']) # query    CLASS N channel w h
        a_class = xs.size(0)
        b_class = xq.size(0)

        n_support = xs.size(1)
        n_query = xq.size(1)
        target_inds = torch.arange(0, a_class).view(a_class, 1, 1).expand(a_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        x = torch.cat([xs.view(a_class * n_support, *xs.size()[2:]),
                       xq.view(b_class * n_query, *xq.size()[2:])], 0)
        z = self.encoder.forward(x)

        z_dim = z.size(-1)

        z_proto = z[:a_class*n_support].view(a_class, n_support, z_dim).mean(1) # shape [60,64]
        zq = z[b_class*n_support:] # shape [300,64]
        dists = euclidean_dist(zq, z_proto) # 欧几里得距离

        # TODO 下一步计算dist，log
        # prey = F.sigmoid(dists)
        log_p_y = F.log_softmax(-dists, dim=1).view(a_class, n_query, -1)  # shape [60 , 5 , 60]
        # gathery = -log_p_y.gather(2, target_inds) 每个样本的logpy值
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)  # y_hat [60 , 5]
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }

@register_model('protonet_conv')
def load_protonet_conv(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, z_dim),
        Flatten()
    )

    return Protonet(encoder)
