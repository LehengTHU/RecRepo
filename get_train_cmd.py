import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                    help='Choose a dataset')
    
    args, _ = parser.parse_known_args()
    return args

args = parse_args()
    
cmds = []
lgn_setting = "--model_name LightGCN --n_layers 2"
mf_setting = "--model_name MF --n_layers 0"
mvae_setting = "--model_name MultVAE --n_layers 0"
cmds.append(f"nohup python main.py --rs_type General --clear_checkpoints --saveID mf --no_wandb --dataset {args.dataset} {mf_setting} --patience 20 --cuda 0 &> logs/{args.dataset}_origin_mf.log &")
cmds.append(f"nohup python main.py --rs_type General --clear_checkpoints --saveID lgn --no_wandb --dataset {args.dataset} {lgn_setting} --patience 20 --cuda 1 &> logs/{args.dataset}_origin_lgn.log &")
cmds.append(f"nohup python main.py --rs_type General --clear_checkpoints --saveID mvae --no_wandb --dataset {args.dataset} {mvae_setting} --patience 20 --cuda 2 &> logs/{args.dataset}_origin_mvae.log &")
cmds.append(f"nohup python main.py --rs_type General --test_only --no_wandb --model_name Pop --dataset {args.dataset} &> logs/{args.dataset}_pop.log &")
# cmds.append(f"python main.py --test_only --no_wandb --model_name Pop --dataset {args.dataset}")
cmds.append(f"python main.py --rs_type General --test_only --no_wandb --model_name Random --dataset {args.dataset}")

with open("training.sh", "w") as f:
    for cmd in cmds:
        f.write(cmd + "\n")
