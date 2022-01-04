import torch.cuda

from pyjssp.simulators import Simulator
import random
import numpy
import time
import torch_geometric
import networkx as nx
from model import to_pyg


def rollout(s, dev):

    s.reset()
    done = False
    # g, r, done = s.observe()
    # for n in g.nodes:
    #     print(n, g.nodes[n])
    p_list = []
    t1 = time.time()
    while True:
        do_op_dict = s.get_doable_ops_in_dict()
        all_machine_work = False if bool(do_op_dict) else True

        # g, _, _ = s.observe()
        # print(list(nx.get_node_attributes(g, name='remain_time').values()))

        if all_machine_work:  # all machines are on processing. keep process!
            s.process_one_time()
        else:  # some of machine has possibly trivial action. the others not.
            _, _, done, sub_list = s.flush_trivial_ops(reward='makespan')
            p_list += sub_list
            if done:
                break  # env rollout finish
            g, r, done = s.observe(return_doable=True)
            g_pre, g_suc, g_dis = to_pyg(g, dev)
            op_id = s.transit()
            p_list.append(op_id)
            g, r, done = s.observe(return_doable=True)
            # for n in g.nodes:
            #     print(n, g.nodes[n])

        if done:
            break  # env rollout finish
    t2 = time.time()
    print('All job finished, makespan={}. Rollout takes {} seconds'.format(s.global_time, t2 - t1))
    return p_list, t2 - t1, s.global_time


if __name__ == "__main__":
    random.seed(0)
    numpy.random.seed(1)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    s = Simulator(6, 6, verbose=False)
    rollout(s, dev)







