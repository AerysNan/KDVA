import numpy as np
import argparse
from replay_trace import replay_trace

parser = argparse.ArgumentParser(description='Replay all traces')
parser.add_argument('--begin', '-b', help='begin framerate', type=int, required=True)
parser.add_argument('--end', '-e', help='end framerate', type=int, required=True)
args = parser.parse_args()


for distill in range(7):
    print(distill)
    for fps in range(args.begin, args.end + 1):
        for batch in [600, 1800]:
            f = open(f'replay_d{distill}_f{fps}-{batch}.csv', 'w')
            mAP_all = np.zeros((9000 // batch + 1, 12), dtype=np.float32)
            for uid in range(1, 13):
                mAP_stream = replay_trace(f'snapshot/result/virat_trace_{uid}_{distill}-{batch}', f'virat_trace_{uid}_{distill}-{batch}', fps, batch)
                mAP_all[:, uid - 1] = mAP_stream
            for row in mAP_all:
                f.write(','.join([str(v) for v in row]))
                f.write('\n')
            f.close()
