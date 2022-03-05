from multiprocessing import Pool
import numpy as np
from evaluate_from_file import evaluate_from_file
import pickle

m = np.zeros((6, 12, 12), dtype=np.double)
m_class = np.zeros((6, 12, 12), dtype=np.double)

p, output = Pool(processes=4), []

for i, dconfig in enumerate(['020', '040', '060', '080', '100']):
    for stream in range(12):
        for epoch in range(11):
            output.append(p.apply_async(evaluate_from_file, (f'snapshot/result/sub_{stream + 1}_{dconfig}_vesna/{epoch + 1:02d}.pkl',
                          f'data/annotations/sub_{stream + 1}_{dconfig if dconfig != "000" else "020"}_val_{epoch}.gt.json',)))

p.close()
p.join()

for i, dconfig in enumerate(['020', '040', '060', '080', '100']):
    for stream in range(12):
        for epoch in range(11):
            result = output[i * 12 * 11 + stream * 11 + epoch].get()
            m[i + 1, stream, epoch + 1] = result["bbox_mAP"]
            m_class[i + 1, stream, epoch + 1] = result["bbox_mAP_car"]

with open('configs/cache/map_distill_val.pkl', 'rb') as f:
    m_base = pickle.load(f)

with open('configs/cache/map_distill_class_val.pkl', 'rb') as f:
    m_base_class = pickle.load(f)

m[0] = m_base[0]
m_class[0] = m_base_class[0]

with open('configs/cache/map_distill_val_esna.pkl', 'wb') as f:
    pickle.dump(m, f)

with open('configs/cache/map_distill_class_val_esna.pkl', 'wb') as f:
    pickle.dump(m_class, f)
