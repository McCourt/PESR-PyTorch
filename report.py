from matplotlib import pyplot as plt
import numpy as np
import parse
import pandas as pd

model_name = 'sesr'
PARSE_STRING = 'Name {} | Epoch {:d} | TotalLoss {:.8f} | DSLoss {:.8f} | L2Loss {:.8f} | psnr {:.8f} | Runtime {:.4f}'
if __name__ == '__main__':
    df = pd.DataFrame(columns=['name', 'epoch', 'TotalLoss', 'DSLoss', 'L2Loss', 'psnr', 'runtime'])
    with open('demo/logs/{}.log'.format(model_name), 'r') as f:
        for id, line in enumerate(f.readlines()):
            parsed = parse.parse(PARSE_STRING, line.strip('\n'))
            df.loc[id] = [i for i in parsed]
    outputs = df.groupby(['epoch'])['psnr'].mean()
    plt.plot(outputs.values)
    plt.title('Model {}'.format(model_name.upper()))
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.savefig('demo/graphs/{}.png'.format(model_name))
