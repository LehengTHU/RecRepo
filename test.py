from parse import parse_args
from utils import fix_seeds
import os
import torch

if __name__ == '__main__':
    args = parse_args()
    print(args)
    # fix_seeds(args.seed)
    rs_type = args.rs_type
    print(f'from models.{rs_type}.'+ args.model_name + ' import ' + args.model_name + '_RS')
    # try:
    #     exec(f'from models.{rs_type}.'+ args.model_name + ' import ' + args.model_name + '_RS') # load the model
    # except:
    #     print('Model %s not implemented!' % (args.model_name))
    
    # exec(f'from models.{rs_type}.'+ args.model_name + ' import ' + args.model_name + '_RS')
    from models.LLM.PerRec import PerRec_RS