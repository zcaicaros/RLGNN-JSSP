import numpy as np
import networkx as nx
from pyjssp import Simulator
from rollout import rollout


def verify_env(n, m):
    s = Simulator(n, m, verbose=False)
    operation_list, _, makespan = rollout(s, 'cpu')
    priority_list = [op_id // n for op_id in operation_list]

    dur_mat, mch_mat = s.processing_time_matrix.astype(np.int32), s.machine_matrix.astype(np.int32)
    n_jobs = mch_mat.shape[0]
    n_machines = mch_mat.shape[1]
    n_operations = n_jobs * n_machines
    assert len(priority_list) == n_jobs * n_machines
    adj_pc = adj_conjunction(n_jobs, n_machines)

    # prepare NIPS adj
    ops_mat = np.arange(0, n_operations).reshape(mch_mat.shape).tolist()  # Init operations mat
    list_for_latest_task_on_machine = [None] * n_machines  # Init list_for_latest_task_on_machine

    adj_mc = np.zeros(shape=[n_operations, n_operations], dtype=int)  # Create adjacent matrix for machine clique
    # Construct NIPS adjacent matrix only for machine cliques
    for job_id in priority_list:
        op_id = ops_mat[job_id][0]
        m_id_for_action = mch_mat[op_id // n_machines, op_id % n_machines]
        if list_for_latest_task_on_machine[m_id_for_action] is not None:
            adj_mc[op_id, list_for_latest_task_on_machine[m_id_for_action]] = 1
        list_for_latest_task_on_machine[m_id_for_action] = op_id
        ops_mat[job_id].pop(0)
    adj_mc = np.pad(adj_mc, ((1, 1), (1, 1)), 'constant', constant_values=0)  # add S and T to machine clique adj
    adj_mc = np.transpose(adj_mc)  # convert input adj from column pointing to row, to, row pointing to column

    adj = adj_pc + adj_mc

    dur_mat = np.pad(dur_mat.reshape(-1, 1), ((1, 1), (0, 0)), 'constant', constant_values=0).repeat(
        n_operations + 2, axis=1)
    edge_weight = np.multiply(adj, dur_mat)
    G = nx.from_numpy_matrix(edge_weight, parallel_edges=False, create_using=nx.DiGraph)  # create nx.DiGraph

    l = nx.dag_longest_path_length(G)
    print('Longest path in the conjunctive graph has length {}.'.format(l))
    if l == makespan:
        print('Length of longest path in conjunctive graph == makespan. Verified')
    else:
        print('Length of longest path in conjunctive graph != makespan. Sth wrong.')


def adj_conjunction(n_job, n_mch):
    adj_mat_pc = np.eye(n_job * n_mch, k=-1, dtype=int)  # Create adjacent matrix for precedence constraints
    adj_mat_pc[np.arange(start=0, stop=n_job * n_mch, step=1).reshape(n_job, -1)[:,
               0]] = 0  # first column does not have upper stream conj_nei
    adj_mat_pc = np.pad(adj_mat_pc, 1, 'constant', constant_values=0)  # pad dummy S and T nodes
    adj_mat_pc[[i for i in range(1, n_job * n_mch + 2 - 1,
                                 n_mch)], 0] = 1  # connect S with 1st operation of each job
    adj_mat_pc[-1, [i for i in range(n_mch, n_job * n_mch + 2 - 1,
                                     n_mch)]] = 1  # connect last operation of each job to T
    adj_mat_pc = np.transpose(adj_mat_pc)  # convert input adj from column pointing to row, to, row pointing to column
    return adj_mat_pc


if __name__ == "__main__":
    np.random.seed(1)

    verify_env(10, 10)





