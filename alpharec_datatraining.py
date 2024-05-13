import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--suffix', nargs='?', default='',
                    help='prefix for dataset')
    parser.add_argument('--lm_model', type=str, default='v3',
                choices=['bert', 'llama2_7b', 'mistral_7b', 'v2', 'v3', 'SFR'],
                help='The base language model')
    parser.add_argument('--model_version', type=str, default='homo',
                choices=['mlp', 'homo'],
                help='The mapping method')
    parser.add_argument('--n_layers', type=int, default=0,
                            help='Number of GCN layers')
    args, _ = parser.parse_known_args()
    return args

args = parse_args()

prompt = ("nohup python main.py --rs_type General --clear_checkpoints "
+ "--saveID <save_id> --dataset <dataset_name> --model_name <model_name> "
+ "--n_layers <n_layer> --patience 20 --cuda <gpu> --no_wandb --train_norm --pred_norm --neg_sample 256 "
+ "--lm_model <lm_model> --model_version <model_version> "
+ "--tau <tau> --infonce 1 &>logs/<log_name>.log &")

# print(prompt)


# dataset = args.dataset
suffix = args.suffix
dataset_list = ['amazon_book_2014', 'amazon_movie', 'amazon_game']
tau_list = [0.15, 0.15, 0.2]
# model_name = 'AgentRerank_uni'
model_name = 'AlphaRec'
lm_model = args.lm_model
model_version = args.model_version

c = 0
n_layer = args.n_layers


cmds = []
for idx, tau in enumerate(tau_list):
    print('tau:', tau)
    dataset = dataset_list[idx]
    save_id = 'tau_' + str(tau) + '_' + lm_model + '_' + model_version + '_' + suffix
    log_name = dataset + '_' + save_id
    filled_prompt = prompt.replace('<model_name>', model_name)
    filled_prompt = filled_prompt.replace('<dataset_name>', dataset)
    filled_prompt = filled_prompt.replace('<lm_model>', lm_model)
    filled_prompt = filled_prompt.replace('<model_version>', model_version)
    filled_prompt = filled_prompt.replace('<gpu>', str(c))
    filled_prompt = filled_prompt.replace('<tau>', str(tau))
    filled_prompt = filled_prompt.replace('<save_id>', save_id)
    filled_prompt = filled_prompt.replace('<log_name>', log_name)
    filled_prompt = filled_prompt.replace('<n_layer>', str(n_layer))
    print(filled_prompt)
    cmds.append(filled_prompt)
    c += 1
    c %= 4
    with open(f"alpharec_total_n_{n_layer}.sh", "w") as f:
        for cmd in cmds:
            f.write(cmd + "\n")
