import torch.cuda
from pyjssp.simulators import Simulator
import random
import numpy
import time
from model import to_pyg, RLGNN


def rollout(s, dev, net=None):

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

        if all_machine_work:  # all machines are on processing. keep process!
            s.process_one_time()
        else:  # some of machine has possibly trivial action. the others not.
            _, _, done, sub_list = s.flush_trivial_ops(reward='makespan')  # flush the trivial action
            p_list += sub_list
            if done:
                break  # env rollout finish
            g, r, done = s.observe(return_doable=True)
            # network forward goes here
            if net is not None:
                g_pre, g_suc, g_dis = to_pyg(g, dev)
                pyg_graphs = {'pre': g_pre, 'suc': g_suc, 'dis': g_dis}
                pyg_graphs = net(**pyg_graphs)
            op_id = s.transit()
            p_list.append(op_id)

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
    net = RLGNN().to(dev)
    rollout(s, dev, net)







