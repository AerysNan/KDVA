import numpy as np
from multiprocessing import Pool
from replay_trace import replay_trace
import pickle
d = {}
p = Pool(processes=6)
with open(f'configs/cache/map_all_golden_na.pkl', 'rb') as f:
    m = pickle.load(f)
with open(f'configs/cache/map_all_class_golden_na.pkl', 'rb') as f:
    m_class = pickle.load(f)
for dconfig in range(6):
    for fconfig in range(5):
        for stream in [12]:
            path = f'snapshot/result/sub_{stream}_{dconfig* 20:03d}'
            if dconfig > 0:
                path += '_na'
            d[(dconfig, fconfig, stream)] = p.apply_async(replay_trace, (path, f'sub_{stream}', (fconfig + 1) * 100, 500,))
p.close()
p.join()
for dconfig in range(6):
    for fconfig in range(5):
        for stream in [12]:
            result = d[(dconfig, fconfig, stream)].get()
            m[dconfig, fconfig, stream-1] = [mAP['bbox_mAP'] for mAP in result]
            m_class[dconfig, fconfig, stream-1] = [mAP['bbox_mAP_car'] for mAP in result]
with open(f'configs/cache/map_all_golden_na.pkl', 'wb') as f:
    pickle.dump(m, f)
with open(f'configs/cache/map_all_class_golden_na.pkl', 'wb') as f:
    pickle.dump(m_class, f)
