from semiMDP.simulators import Simulator
import torch
import random
import numpy as np
import networkx as nx
from torch.nn.functional import relu, softmax
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch.distributions.categorical import Categorical


def count_parameters(model, verbose=False, print_model=False):
    """
    model: torch nn
    """
    if print_model:
        print('Model:', model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)

    print('The model has {} parameters'.format(pytorch_total_params))


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


class RLGNNLayer(MessagePassing):
    def __init__(self,
                 num_mlp_layer=2,
                 in_chnl=8,
                 hidden_chnl=256,
                 out_chnl=8):
        super(RLGNNLayer, self).__init__()

        self.module_pre = MLP(num_layers=num_mlp_layer, in_chnl=in_chnl, hidden_chnl=hidden_chnl, out_chnl=out_chnl)
        self.module_suc = MLP(num_layers=num_mlp_layer, in_chnl=in_chnl, hidden_chnl=hidden_chnl, out_chnl=out_chnl)
        self.module_dis = MLP(num_layers=num_mlp_layer, in_chnl=in_chnl, hidden_chnl=hidden_chnl, out_chnl=out_chnl)
        self.module_merge = MLP(num_layers=num_mlp_layer, in_chnl=6*out_chnl, hidden_chnl=hidden_chnl, out_chnl=out_chnl)

    def reset_parameters(self):
        reset(self.module_pre)
        reset(self.module_suc)
        reset(self.module_dis)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def forward(self, raw_feature, **graphs):

        graph_pre = graphs['pre']
        graph_suc = graphs['suc']
        graph_dis = graphs['dis']

        num_nodes = graph_pre.num_nodes  # either pre, suc, or dis will work
        h_before_process = graph_pre.x  # either pre, suc, or dis will work

        # message passing
        out_pre = self.propagate(graph_pre.edge_index, x=graph_pre.x, size=None)
        out_suc = self.propagate(graph_suc.edge_index, x=graph_suc.x, size=None)
        out_dis = self.propagate(graph_dis.edge_index, x=graph_dis.x, size=None)

        # process aggregated messages
        out_pre = self.module_pre(out_pre)
        out_suc = self.module_suc(out_suc)
        out_dis = self.module_dis(out_dis)

        # merge different h
        h = torch.cat([relu(out_pre),
                       relu(out_suc),
                       relu(out_dis),
                       relu(h_before_process.sum(dim=0).tile(num_nodes, 1)),
                       h_before_process,
                       raw_feature], dim=1)
        h = self.module_merge(h)

        # new graphs after processed by this layer
        graph_pre = Data(x=h, edge_index=graph_pre.edge_index)
        graph_suc = Data(x=h, edge_index=graph_suc.edge_index)
        graph_dis = Data(x=h, edge_index=graph_dis.edge_index)

        return {'pre': graph_pre, 'suc': graph_suc, 'dis': graph_dis}


class RLGNN(torch.nn.Module):
    def __init__(self,
                 num_mlp_layer=2,
                 num_layer=3,
                 in_chnl=8,
                 hidden_chnl=256,
                 out_chnl=8):
        super(RLGNN, self).__init__()

        self.layers = torch.nn.ModuleList()

        for l in range(num_layer):
            if l == 0:  # initial layer
                self.layers.append(RLGNNLayer(num_mlp_layer=num_mlp_layer,
                                              in_chnl=in_chnl,
                                              hidden_chnl=hidden_chnl,
                                              out_chnl=out_chnl))
            else:  # the rest layers
                self.layers.append(RLGNNLayer(num_mlp_layer=num_mlp_layer,
                                              in_chnl=out_chnl,
                                              hidden_chnl=hidden_chnl,
                                              out_chnl=out_chnl))

    def forward(self, raw_feature, **graphs):
        for layer in self.layers:
            graphs = layer(raw_feature, **graphs)
        return graphs


class PolicyNet(torch.nn.Module):
    def __init__(self,
                 num_mlp_layer=2,
                 in_chnl=8,
                 hidden_chnl=256,
                 out_chnl=1):
        super(PolicyNet, self).__init__()

        self.policy = MLP(num_layers=num_mlp_layer, in_chnl=in_chnl, hidden_chnl=hidden_chnl, out_chnl=out_chnl)

    def forward(self, node_h, feasible_op_id):
        logit = self.policy(node_h).view(-1)
        pi = softmax(logit[feasible_op_id], dim=0)
        dist = Categorical(probs=pi)
        sampled_op_id = dist.sample()
        sampled_op = feasible_op_id[dist.sample().item()]
        log_prob = dist.log_prob(sampled_op_id)
        return sampled_op, log_prob


class CriticNet(torch.nn.Module):
    def __init__(self,
                 num_mlp_layer=2,
                 in_chnl=8,
                 hidden_chnl=256,
                 out_chnl=1):
        super(CriticNet, self).__init__()

        self.critic = MLP(num_layers=num_mlp_layer, in_chnl=in_chnl, hidden_chnl=hidden_chnl, out_chnl=out_chnl)

    def forward(self, node_h):
        v = self.critic(node_h.sum(dim=0))
        return v


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(1)
    torch.manual_seed(1)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    # dev = 'cpu'

    s = Simulator(3, 3, verbose=False)
    print(s.machine_matrix)
    print(s.processing_time_matrix)
    s.reset()

    g, r, done = s.observe()

    g_pre, g_suc, g_dis = to_pyg(g, dev)
    # raw feature
    raw_feature = g_pre.x

    # test mlp
    mlp = MLP().to(dev)
    # count_parameters(mlp)
    out = mlp(g_pre.x)
    mlp_grad = torch.autograd.grad(out.mean(), [param for param in mlp.parameters()])

    # test rlgnn_layer
    rlgnn_layer = RLGNNLayer().to(dev)
    # count_parameters(rlgnn_layer)
    new_graphs = rlgnn_layer(raw_feature, **{'pre': g_pre, 'suc': g_suc, 'dis': g_dis})
    loss = sum([pyg.x.mean() for pyg in new_graphs.values()])
    rlgnn_layer_grad = torch.autograd.grad(loss, [param for param in rlgnn_layer.parameters()])

    # test rlgnn net
    net = RLGNN().to(dev)
    # count_parameters(net)
    new_graphs = net(raw_feature, **{'pre': g_pre, 'suc': g_suc, 'dis': g_dis})
    loss = sum([pyg.x.mean() for pyg in new_graphs.values()])
    rlgnn_grad = torch.autograd.grad(loss, [param for param in net.parameters()])

    # test policy net
    policy = PolicyNet().to(dev)
    # count_parameters(policy)
    _, log_p = policy(new_graphs['pre'].x, s.get_doable_ops_in_list())
    policy_grad = torch.autograd.grad(log_p, [param for param in policy.parameters()])

    # test critic net
    critic = CriticNet().to(dev)
    # count_parameters(critic)
    v = critic(new_graphs['pre'].x)
    critic_grad = torch.autograd.grad(v, [param for param in critic.parameters()])

