from pyjssp.simulators import Simulator
import torch
import random
import numpy as np
import time
import torch_geometric
from torch_geometric.data import Data
import networkx as nx


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


class RLGNNLayer(torch.nn.Module):
    def __init__(self, in_chnl, out_chnl):
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

    print(g_pre.x)