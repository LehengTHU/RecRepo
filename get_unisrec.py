import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                    help='Choose a dataset')
    parser.add_argument('--cuda', type=int, default=0,
                        help='cuda device.')
    
    args, _ = parser.parse_known_args()
    return args

args = parse_args()
Ks = 20
    
cmds = []
cmds.append(f"nohup python main.py --rs_type General --model_name UniSRec --clear_checkpoints --saveID UniSRec --no_wandb --dataset {args.dataset} --patience 20 --cuda {args.cuda} --Ks {Ks} &> logs/{args.dataset}_UniSRec.log &")

with open("unisrec_training.sh", "w") as f:
    for cmd in cmds:
        f.write(cmd + "\n")
