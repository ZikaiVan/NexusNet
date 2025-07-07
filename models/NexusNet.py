import numpy as np
import torch
from torch import nn
from models.utils import glorot_weight_zero_bias, Expression, np_to_var, normalize_adj
from abc import abstractmethod
from . import algos


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, inputs):
        """
        Forward pass logic
        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1., **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm is not None:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0,
                                         maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=1., **kwargs):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm is not None:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0,
                                         maxnorm=self.max_norm)
        return super(LinearWithConstraint, self).forward(x)


def _transpose_to_0312(x):
    return x.permute(0, 3, 1, 2)


def _transpose_to_0132(x):
    return x.permute(0, 1, 3, 2)


def _review(x):
    return x.contiguous().view(-1, x.size(2), x.size(3))


def _squeeze_final_output(x):
    """
    Remove empty dim at end and potentially remove empty time dim
    Do not just use squeeze as we never want to remove first dim
    :param x:
    :return:
    """
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x


class Nexus(nn.Module):
    def __init__(self,
                 flag,
                 n_nodes,
                 input_dim,
                 adj,
                 gadj,
                 dataset,
                 k=1,):
        super(Nexus, self).__init__()
        self.__dict__.update(locals())
        del self.self

        self.xs, self.ys = torch.tril_indices(self.n_nodes, self.n_nodes, offset=-1)
        node_value = adj[self.xs, self.ys] 
        self.edge_weight = nn.Parameter(node_value.clone().detach())

        q_factor = 5
        self.shortest_dist, self.path = algos.floyd_warshall(np.ceil(np.array(gadj) / q_factor) * q_factor)
        self.max_dist = int(np.amax(self.shortest_dist)) + 1
        self.spatial_pos = torch.from_numpy((self.shortest_dist)).long()
        self.spatial_pos_encoder = nn.Embedding(self.max_dist, 1, padding_idx=0)

        region = np.zeros(self.n_nodes)        
        index = 0
        if dataset == 'BNCI2014001':            
            region[0] = 0
            region[1:6] = 1
            region[6:13] = 2
            region[13:18] = 3
            region[18:21] = 4
            region[21] = 5
            edge_attr = np.zeros((21, 3))
            for i in range(6):
                for j in range(i, 6):
                    edge_attr[index] = np.array([i!=j, i, j])
                    index += 1
        elif dataset == 'BNCI2014004':
            region[0] = 0
            edge_attr = np.zeros((1, 3))
            for i in range(1):
                for j in range(i, 1):
                    edge_attr[index] = np.array([i!=j, i, j])
                    index += 1

        route = algos.gen_edge_input(self.max_dist, self.path, edge_attr, region)
        self.route = torch.from_numpy(route).long()
        self.edge_dis_encoder = nn.Embedding(self.n_nodes**2, 1)
        self.edge_encoder = nn.Embedding(100, 1, padding_idx=0)


    def forward(self, x):
    # Check weight mode
        if all(v == 0 for v in self.flag):
            return x
        
        neighbor = torch.zeros([self.n_nodes, self.n_nodes], device=x.device)
        neighbor[self.xs.to(x.device), self.ys.to(x.device)] = self.edge_weight.to(x.device)
        neighbor = neighbor + neighbor.T + torch.eye(self.n_nodes, dtype=neighbor.dtype, device=x.device)
        neighbor = normalize_adj(neighbor, mode='sym')

    # Spatial Nexus Forward
        spatial = self.spatial_pos_encoder(self.spatial_pos.to(x.device)).squeeze()

    # Route Nexus Forward
        route = self.edge_encoder(self.route.to(x.device)).mean(-2)
        max_hop = route.size(-2) 
        route_flat = route.permute(2, 0, 1, 3).reshape(
                max_hop, -1, 1
            )
        route_flat = torch.bmm(
                route_flat,
                self.edge_dis_encoder.weight.reshape(
                    -1, 1, 1
                )[:max_hop, :, :],
            )
        route = route_flat.reshape(
                max_hop, self.n_nodes, self.n_nodes, 1
            ).permute(1, 2, 0, 3)
        route = (
                route.sum(-2) 
            ).squeeze()
        bias = torch.softmax(self.flag[0]*neighbor + self.flag[1]*spatial + self.flag[2]*route, dim=1)
        x_edge = torch.matmul(bias.unsqueeze(0), x)

        return x_edge


class NexusNet(BaseModel):
    def __init__(self,
                 flag,
                 Adj,
                 eu_adj,
                 centrality,
                 in_chans,
                 n_classes,
                 final_conv_length='auto',
                 input_time_length=None,
                 pool_mode='mean',
                 f1=8,
                 d=2,
                 f2=16, 
                 kernel_length=64,
                 third_kernel_size=(8, 4),
                 drop_prob=0.25,
                 dataset=None,
                 ):
        super(NexusNet, self).__init__()
        
        if final_conv_length == 'auto':
            assert input_time_length is not None

        self.__dict__.update(locals())
        del self.self
        self.flag = [torch.tensor(x, dtype=torch.float32) for x in self.flag]
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]

        self.temporal_conv = nn.Sequential(
            Expression(_transpose_to_0312),
            Conv2dWithConstraint(in_channels=1, out_channels=self.f1,
                                 kernel_size=(1, self.kernel_length),
                                 max_norm=None,
                                 stride=1,
                                 bias=False,
                                 padding=(0, self.kernel_length // 2)
                                 ),
            nn.BatchNorm2d(self.f1, momentum=0.01, affine=True, eps=1e-3),
        )

        self.centrality_encoder = nn.Embedding(torch.max(self.centrality)+1, 1)
        self.nexus = nn.Sequential(
            Expression(_review),
            Nexus(self.flag[0:3], self.in_chans, self.input_time_length, adj=self.Adj, gadj=self.eu_adj, dataset=self.dataset),
        )

        self.spatial_conv = nn.Sequential(
            Conv2dWithConstraint(self.f1, self.f1 * self.d, (self.in_chans, 1),
                                 max_norm=1, stride=1, bias=False,
                                 groups=self.f1, padding=(0, 0)),
            nn.BatchNorm2d(self.f1 * self.d, momentum=0.01, affine=True,
                           eps=1e-3),
            nn.ELU(),
            pool_class(kernel_size=(1, 8), stride=(1, 8))
        )

        self.separable_conv = nn.Sequential(
            nn.Dropout(p=self.drop_prob),
            Conv2dWithConstraint(self.f1 * self.d, self.f1 * self.d, (1, 16),
                                 max_norm=None,
                                 stride=1,
                                 bias=False, groups=self.f1 * self.d,
                                 padding=(0, 8)),
            Conv2dWithConstraint(self.f1 * self.d, self.f2, (1, 1), max_norm=None, stride=1, bias=False,
                                 padding=(0, 0)),
            nn.BatchNorm2d(self.f2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            pool_class(kernel_size=(1, 8), stride=(1, 8))
        )

        out = np_to_var(np.ones((1, self.in_chans, self.input_time_length, 1), dtype=np.float32))
        out = self.forward_init(out)
        n_out_virtual_chans = out.cpu().data.numpy().shape[2]

        if self.final_conv_length == 'auto':
            n_out_time = out.cpu().data.numpy().shape[3]
            self.final_conv_length = n_out_time

        self.cls = nn.Sequential(
            nn.Dropout(p=self.drop_prob),
            Conv2dWithConstraint(self.f2, self.n_classes,
                                 (n_out_virtual_chans, self.final_conv_length), max_norm=0.25,
                                 bias=True),
            Expression(_transpose_to_0132),
            Expression(_squeeze_final_output)
        )

        self.apply(glorot_weight_zero_bias)

    def forward_init(self, x):
        with torch.no_grad():
            centrality_bias = self.centrality_encoder(self.centrality.to(x.device))
            batch_size = x.size(0)
            x = self.temporal_conv(x)
            x = self.nexus(x)
            x = x.view(1, batch_size, -1, x.size(-2), x.size(-1))
            x = x.permute(1, 2, 0, 3, 4).contiguous().view(batch_size, -1, x.size(-2), x.size(-1))
            # Centrality Nexus Forward
            if (self.flag[3]):
                centrality_bias = self.centrality_encoder(self.centrality.to(x.device)).view(1, 1, self.centrality.shape[0], 1)
                centrality_bias = torch.softmax(centrality_bias, dim=2)
                x= x*centrality_bias
            x = self.spatial_conv(x)
            x = self.separable_conv(x)
        return x

    def forward(self, inputs):
        return self.forward_once(inputs)

    def forward_once(self, x):
        feature = []
        batch_size = x.size(0)
        x = x[:, :, :, None]
        x = self.temporal_conv(x)
        x = self.nexus(x)        
        x = x.view(1, batch_size, -1, x.size(-2), x.size(-1))
        x = x.permute(1, 0, 2, 3, 4).contiguous().view(batch_size, -1, x.size(-2), x.size(-1))
        # Centrality Nexus Forward
        if self.flag[3]:
            centrality_bias = self.centrality_encoder(self.centrality.to(x.device)).view(1, 1, self.centrality.shape[0], 1)
            centrality_bias = torch.softmax(centrality_bias, dim=2)
            x = x*centrality_bias
        x = self.spatial_conv(x)
        x = self.separable_conv(x)
        x = self.cls(x)
        return x, feature

