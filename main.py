from parse import parse_args
from utils import fix_seeds
import os
import torch

from models.Seq.SASRec import SASRec
# from models.LLM.SASRecModules_ori import *

if __name__ == '__main__':
    args = parse_args()

    fix_seeds(args.seed) # set random seed

    rs_type = args.rs_type # LLM, Seq, General, etc.
    print(f'from models.{rs_type}.'+ args.model_name + ' import ' + args.model_name + '_RS')

    try:
        exec(f'from models.{args.rs_type}.'+ args.model_name + ' import ' + args.model_name + '_RS') # load the model
    except:
        print('Model %s not implemented!' % (args.model_name))
    
    RS = eval(args.model_name + '_RS(args)') # load the recommender system
    
    RS.execute() # train and test
    print('Done!')

