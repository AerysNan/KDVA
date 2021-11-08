import pickle
import argparse

parser = argparse.ArgumentParser(
    description='Merge multiple result file')
parser.add_argument(
    '--path', '-p', help='path to result files', type=str, required=True)
parser.add_argument(
    '--count', '-c', help='number of result files', type=int, required=True)
args = parser.parse_args()

result = []
for i in range(args.count):
    with open(f'{args.path}/result_{i}.pkl', 'rb') as f:
        o = pickle.load(f)
        result += o

with open(f'{args.path}/result.pkl', 'wb') as f:
    pickle.dump(result, f)
