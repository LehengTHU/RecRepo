import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                    help='Choose a dataset')
    
    args, _ = parser.parse_known_args()
    return args

args = parse_args()

prompt = ("nohup python main.py --rs_type General --clear_checkpoints "
+ "--saveID <save_id> --dataset <dataset_name> --model_name <model_name> "
+ "--n_layers <n_layer> --patience 20 --cuda <gpu> --no_wandb "
+ "--temp_cl <tau> --eps_XSimGCL <eps> --lambda_cl <lambda> &>logs/<log_name>.log &")

print(prompt)

# cmds = []

# dataset_list = ['ml-1m', 'amazon_book_2014', 'steam', 'mix_movie_book']
dataset = args.dataset
tau_list = [0.05, 0.1, 0.15, 0.2]
eps_list = [0.01, 0.05, 0.1, 1]
lam_list = [0.01, 0.05, 0.1, 1]
# model_name = 'AgentRerank_uni'
model_name = 'KAR'

c = 0
n_layer = 2


cmds = []
for tau in tau_list:
    print('tau:', tau)
    save_id = 'tau_' + str(tau) + '_eps_' + str(eps_list[0]) + '_lam_' + str(lam_list[0])
    log_name = dataset + '_' + save_id + '_' + model_name
    filled_prompt = prompt.replace('<model_name>', model_name)
    filled_prompt = filled_prompt.replace('<dataset_name>', dataset)
    filled_prompt = filled_prompt.replace('<gpu>', str(c))
    filled_prompt = filled_prompt.replace('<tau>', str(tau))
    filled_prompt = filled_prompt.replace('<eps>', str(eps_list[0]))
    filled_prompt = filled_prompt.replace('<lambda>', str(lam_list[0]))
    filled_prompt = filled_prompt.replace('<save_id>', save_id)
    filled_prompt = filled_prompt.replace('<log_name>', log_name)
    filled_prompt = filled_prompt.replace('<n_layer>', str(n_layer))
    print(filled_prompt)
    cmds.append(filled_prompt)
    c += 1
    c %= 4
    with open(f"training_{model_name}_n_{n_layer}.sh", "w") as f:
        for cmd in cmds:
            f.write(cmd + "\n")

for eps in eps_list[1:]:
    print('eps:', eps)
    save_id = 'tau_' + str(tau_list[0]) + '_eps_' + str(eps) + '_lam_' + str(lam_list[0])
    log_name = dataset + '_' + save_id + '_' + model_name
    filled_prompt = prompt.replace('<model_name>', model_name)
    filled_prompt = filled_prompt.replace('<dataset_name>', dataset)
    filled_prompt = filled_prompt.replace('<gpu>', str(c))
    filled_prompt = filled_prompt.replace('<tau>', str(tau_list[0]))
    filled_prompt = filled_prompt.replace('<eps>', str(eps))
    filled_prompt = filled_prompt.replace('<lambda>', str(lam_list[0]))
    filled_prompt = filled_prompt.replace('<eps>', str(eps))
    filled_prompt = filled_prompt.replace('<save_id>', save_id)
    filled_prompt = filled_prompt.replace('<log_name>', log_name)
    filled_prompt = filled_prompt.replace('<n_layer>', str(n_layer))
    print(filled_prompt)
    cmds.append(filled_prompt)
    c += 1
    c %= 4
    with open(f"training_{model_name}_n_{n_layer}.sh", "w") as f:
        for cmd in cmds:
            f.write(cmd + "\n")

for lam in lam_list[1:]:
    print('lambda:', lam)
    save_id = 'tau_' + str(tau_list[0]) + '_eps_' + str(eps_list[0]) + '_lam_' + str(lam)
    log_name = dataset + '_' + save_id + '_' + model_name
    filled_prompt = prompt.replace('<model_name>', model_name)
    filled_prompt = filled_prompt.replace('<dataset_name>', dataset)
    filled_prompt = filled_prompt.replace('<gpu>', str(c))
    filled_prompt = filled_prompt.replace('<tau>', str(tau_list[0]))
    filled_prompt = filled_prompt.replace('<eps>', str(eps_list[0]))
    filled_prompt = filled_prompt.replace('<lambda>', str(lam))
    filled_prompt = filled_prompt.replace('<save_id>', save_id)
    filled_prompt = filled_prompt.replace('<log_name>', log_name)
    filled_prompt = filled_prompt.replace('<n_layer>', str(n_layer))
    print(filled_prompt)
    cmds.append(filled_prompt)
    c += 1
    c %= 4
    with open(f"training_{model_name}_n_{n_layer}.sh", "w") as f:
        for cmd in cmds:
            f.write(cmd + "\n")