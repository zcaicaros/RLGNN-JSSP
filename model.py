from pyjssp.simulators import Simulator
import torch
import random
import numpy as np
import time
import torch_geometric
from torch_geometric.data import Data
import networkx as nx


def count_parameters(model, verbose=False):
    """
    model: torch nn
    """
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
    print('Model:', model, 'has {} parameters'.format(pytorch_total_params))


def to_pyg(g, dev):

    x = []
    one_hot = np.eye(3, dtype=np.float32)[np.fromiter(nx.get_node_attributes(g, 'type').values(), dtype=np.int32)]
    x.append(one_hot)
    x.append(np.fromiter(nx.get_node_attributes(g, 'processing_time').values(), dtype=np.float32).reshape(-1, 1))
    x.append(np.fromiter(nx.get_node_attributes(g, 'complete_ratio').values(), dtype=np.float32).reshape(-1, 1))
    x.append(np.fromiter(nx.get_node_attributes(g, 'remaining_ops').values(), dtype=np.float32).reshape(-1, 1))
    x.append(np.fromiter(nx.get_node_attributes(g, 'waiting_time').values(), dtype=np.float32).reshape(-1, 1))
    x.append(np.fromiter(nx.get_node_attributes(g, 'remain_time').values(), dtype=np.float32).reshape(-1, 1))
    x = np.concatenate(x, axis=1)
    x = torch.from_numpy(x)

    for n in g.nodes:
        if g.nodes[n]['type'] == 1:
            x[n] = 0  # finished op has feature 0

    adj_pre = np.zeros([g.number_of_nodes(), g.number_of_nodes()], dtype=np.float32)
    adj_suc = np.zeros([g.number_of_nodes(), g.number_of_nodes()], dtype=np.float32)
    adj_dis = np.zeros([g.number_of_nodes(), g.number_of_nodes()], dtype=np.float32)
    for e in g.edges:
        s, t = e
        if g.nodes[s]['id'][0] == g.nodes[t]['id'][0]:  # conjunctive edge
            if g.nodes[s]['id'][1] < g.nodes[t]['id'][1]:  # forward
                adj_pre[s, t] = 1
            else:  # backward
                adj_suc[s, t] = 1
        else:  # disjunctive edge
            adj_dis[s, t] = 1
    edge_index_pre = torch.nonzero(torch.from_numpy(adj_pre)).t().contiguous()
    edge_index_suc = torch.nonzero(torch.from_numpy(adj_suc)).t().contiguous()
    edge_index_dis = torch.nonzero(torch.from_numpy(adj_dis)).t().contiguous()

    g_pre = Data(x=x, edge_index=edge_index_pre).to(dev)
    g_suc = Data(x=x, edge_index=edge_index_suc).to(dev)
    g_dis = Data(x=x, edge_index=edge_index_dis).to(dev)

    return g_pre, g_suc, g_dis


class MLP(torch.nn.Module):
    def __init__(self,
                 num_layers=2,
                 in_chnl=8,
                 hidden_chnl=256,
                 out_chnl=8):
        super(MLP, self).__init__()

        self.layers = torch.nn.ModuleList()

        for l in range(num_layers):
            if l == 0:  # first layer
                self.layers.append(torch.nn.Linear(in_chnl, hidden_chnl))
                self.layers.append(torch.nn.ReLU())
                if num_layers == 1:
                    self.layers.append(torch.nn.Linear(hidden_chnl, out_chnl))
            elif l <= num_layers - 2:  # hidden layers
                self.layers.append(torch.nn.Linear(hidden_chnl, hidden_chnl))
                self.layers.append(torch.nn.ReLU())
            else:  # last layer
                self.layers.append(torch.nn.Linear(hidden_chnl, hidden_chnl))
                self.layers.append(torch.nn.ReLU())
                self.layers.append(torch.nn.Linear(hidden_chnl, out_chnl))

    def forward(self, h):
        for lyr in self.layers:
            h = lyr(h)
        return h


class RLGNNLayer(torch.nn.Module):
    def __init__(self,
                 num_layers,
                 in_chnl,
                 hidden_chnl,
                 out_chnl):
        super(RLGNNLayer, self).__init__()


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(1)

    # dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = 'cpu'

    s = Simulator(3, 3, verbose=False)
    print(s.machine_matrix)
    print(s.processing_time_matrix)
    s.reset()

    g, r, done = s.observe()

    g_pre, g_suc, g_dis = to_pyg(g, dev)

    mlp = MLP().to(dev)
    # count_parameters(mlp)
    out = mlp(g_pre.x)
    mlp_grad = torch.autograd.grad(out.mean(), [param for param in mlp.parameters()])


