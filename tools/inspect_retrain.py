from fake_distill import allocate
import numpy as np
import pickle

m, m_class = {}, {}
keys = ['gt', 'val', 'val_na', 'val_es', 'val_esna', 'golden', 'golden_na', 'golden_es', 'golden_esna']
for key in keys:
    with open(f'configs/cache/map_distill_{key}.pkl', 'rb') as f:
        m[key] = pickle.load(f)
        if 'golden' in key:
            m[key] = m[key]
    with open(f'configs/cache/map_distill_class_{key}.pkl', 'rb') as f:
        m_class[key] = pickle.load(f)
        if 'golden' in key:
            m_class[key] = m_class[key]


def compare(k1, k2, classwise):
    l, m_base = [], m if not classwise else m_class
    for i in range(12):
        if k1 == k2:
            l.append(np.corrcoef(m_base[k1][:, i, 1:].reshape(1, -1), m_base[k2][:, i, :-1].reshape(1, -1))[0, 1])
        else:
            l.append(np.corrcoef(m_base[k1][:, i, 2:].reshape(1, -1), m_base[k2][:, i, 1: -1].reshape(1, -1))[0, 1])
    return sum(l) / len(l)


def generate_plan(k, throughput, n_stream, classwise):
    optimal_plan, aca_plan,  m_base = np.zeros((12, n_stream), dtype=np.int64), np.zeros((12, n_stream), dtype=np.int64), m if not classwise else m_class
    for epoch in range(12):
        optimal_plan[epoch, :] = allocate(throughput * n_stream, m_base[f'golden_{k}'][:, :n_stream, epoch])
        aca_plan[epoch, :] = throughput
        if epoch > 1:
            m_observe = m_base[f'val_{k}'][:, :n_stream, epoch - 1]
            for i in range(n_stream):
                for j in range(n_stream):
                    if i == j:
                        continue
                    while True:
                        if aca_plan[epoch, i] == 5 or aca_plan[epoch, j] == 0:
                            break
                        current_map = m_observe[aca_plan[epoch, i], i] + m_observe[aca_plan[epoch, j], j]
                        updated_map = m_observe[aca_plan[epoch, i] + 1, i] + m_observe[aca_plan[epoch, j] - 1, j]
                        if current_map > updated_map:
                            break
                        aca_plan[epoch, i] += 1
                        aca_plan[epoch, j] -= 1
    return optimal_plan, aca_plan
