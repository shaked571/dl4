import argparse

from main import Trainer
from os import listdir
from os.path import isfile, join

parser = argparse.ArgumentParser(description='Aligner model')
parser.add_argument('-ld', '--lstm_drop', help='lstm dropout', action='store_true')
parser.add_argument('-le', '--lstm_embedding', help='embedding dropout', action='store_true')
parser.add_argument('-x', '--xavier', help='to use xavier init', action='store_true')
parser.add_argument('-a', '--adamw', help='to use adamW instead of RMSprop ', action='store_true')
parser.add_argument('-e', '--epoch', help='embedding dropout', type=int,default=3, required=False)
parser.add_argument('-s', '--seed', help='seed', type=int,default=1, required=False)
parser.add_argument('-l', '--lr', help='learning rate', type=float,default=0.001, required=False)
parser.add_argument('-b', '--batch', help='train batch size', type=int,default=128, required=False)
parser.add_argument('-do', '--drop', help='train batch size', type=float,default=0.25, required=False)
parser.add_argument('--gpu', type=int, default=0, required=False)
args = parser.parse_args()
print(args.lr)
trainer = Trainer(args.lstm_drop, args.lstm_embedding, args.xavier, args.adamw,n_ep=args.epoch,
                  gpu=args.gpu,batch=args.batch, lr=args.lr, seed=args.seed, dropout=args.drop)
mypath = 'outputs'
models = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

for m in models:
    print(m)
    trainer.test(m)
