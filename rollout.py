import numpy as np
import torch
from pyjssp.simulators import Simulator
import random
import numpy
import time
from model import to_pyg, RLGNN, PolicyNet, CriticNet


def rollout(s, dev, embedding_net=None, policy_net=None, critic_net=None, verbose=True):

    if embedding_net is not None and \
            policy_net is not None and \
            critic_net is not None:
        embedding_net.to(dev)
        policy_net.to(dev)
        critic_net.to(dev)

    s.reset()
    done = False

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
            if embedding_net is not None and \
                    policy_net is not None and \
                    critic_net is not None:  # network forward goes here
                g_pre, g_suc, g_dis = to_pyg(g, dev)
                raw_feature = g_pre.x  # either pre, suc, or dis will work
                pyg_graphs = {'pre': g_pre, 'suc': g_suc, 'dis': g_dis}
                pyg_graphs = embedding_net(raw_feature, **pyg_graphs)
                feasible_op_id = s.get_doable_ops_in_list()
                sampled_action, _ = policy_net(pyg_graphs['pre'].x, feasible_op_id)  # either pre, suc, or dis will work
                s.transit(sampled_action)
                p_list.append(sampled_action)
                v = critic_net(pyg_graphs['pre'].x)  # either pre, suc, or dis will work
            else:
                op_id = s.transit()
                p_list.append(op_id)

        if done:
            break  # env rollout finish
    t2 = time.time()
    if verbose:
        print('All job finished, makespan={}. Rollout takes {} seconds'.format(s.global_time, t2 - t1))
    return p_list, t2 - t1, s.global_time


if __name__ == "__main__":
    random.seed(0)
    numpy.random.seed(1)
    torch.manual_seed(1)

    setting = 'm=5'  # 'm=5', 'j=30', 'free_for_all'

    if setting == 'm=5':
        j = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        m = [5 for _ in range(len(j))]
    elif setting == 'j=30':
        m = [5, 10, 15, 20, 25, 30]
        j = [30 for _ in range(len(m))]
    else:
        m = [5]
        j = [30]
    save_dir = 'plt/RL-GNN_complexity_{}_reimplement.npy'.format(setting)

    embed = RLGNN()
    policy = PolicyNet()
    critic = CriticNet()
    print('Warm start...')
    for p_m, p_j in zip([5], [5]):  # select problem size
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        # dev = 'cpu'
        s = Simulator(p_m, p_j, verbose=False)
        _, t, _ = rollout(s, dev, embed, policy, critic, verbose=False)
    times = []
    for p_m, p_j in zip(m, j):  # select problem size
        print('Problem size = (m={}, j={})'.format(p_m, p_j))
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        # dev = 'cpu'
        s = Simulator(p_m, p_j, verbose=False)
        _, t, _ = rollout(s, dev, embed, policy, critic)
        times.append(t)

    # print(times)

    numpy.save(save_dir, np.array(times))

