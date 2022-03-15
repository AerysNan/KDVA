from multiprocessing import Pool
import numpy as np
import math
from evaluate_from_file import evaluate_from_file
import pickle

# m = np.zeros((6, 12, 15), dtype=np.double)
# m_class = np.zeros((6, 12, 15), dtype=np.double)

# p, output = Pool(processes=4), []

# for i, dconfig in enumerate(['000', '020', '040', '060', '080', '100']):
#     for stream in range(12):
#         for epoch in range(14):
#             output.append(p.apply_async(evaluate_from_file, (f'snapshot/result/virat_{stream + 1}_{dconfig}_vbase/{epoch + 1:02d}.pkl',
#                           f'data/annotations/virat_{stream + 1}_{dconfig if dconfig != "000" else "020"}_val_{epoch}.golden.json',)))

# p.close()
# p.join()

# for i, dconfig in enumerate(['000', '020', '040', '060', '080', '100']):
#     for stream in range(12):
#         for epoch in range(14):
#             result = output[i * 12 * 11 + stream * 11 + epoch].get()
#             m[i, stream, epoch + 1] = result["bbox_mAP"]
#             classes_of_interest = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
#             mAPs_classwise = [result["classwise"][c] for c in classes_of_interest if not math.isnan(result["classwise"][c])]
#             m_class[i, stream, epoch + 1] = sum(mAPs_classwise) / len(mAPs_classwise)


# with open('configs/cache/virat_distill_val.pkl', 'wb') as f:
#     pickle.dump(m, f)

# with open('configs/cache/virat_distill_class_val.pkl', 'wb') as f:
#     pickle.dump(m_class, f)


with open('configs/cache/virat_distill_class.pkl', 'rb') as f:
    m_class = pickle.load(f)
p, output = Pool(processes=4), {}

for i, dconfig in enumerate(['020', '040', '060', '080', '100']):
    for stream in range(12):
        output[(i, stream)] = p.apply_async(evaluate_from_file, (f'snapshot/merge/virat_{stream+ 1}_{dconfig}_base.pkl',
                                                                 f'data/annotations/virat_{stream + 1}.golden.json',))

p.close()
p.join()

for i in range(5):
    for stream in range(12):
        result = output[(i, stream)].get()
        classes_of_interest = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
        mAPs_classwise = [result["classwise"][c] for c in classes_of_interest if not math.isnan(result["classwise"][c])]
        m_class[i + 1, stream, -1] = sum(mAPs_classwise) / len(mAPs_classwise)


# with open('configs/cache/virat_distill_val.pkl', 'wb') as f:
#     pickle.dump(m, f)

# with open('configs/cache/virat_distill_class_val.pkl', 'wb') as f:
#     pickle.dump(m_class, f)
