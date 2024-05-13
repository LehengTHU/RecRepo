#%%
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import torch
import numpy as np
import os
import os.path as op
from tqdm import tqdm


os.environ['HF_HOME'] 
#%%
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
model_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
model = AutoModelForCausalLM.from_pretrained(model_path, device_map = 'auto',token='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')
tokenizer = AutoTokenizer.from_pretrained(model_path, device_map = 'auto',token='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')
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
item_df = pd.read_csv('raw_data/items_filtered.csv', index_col=0)
item_df.rename(columns={'title': 'item_name'}, inplace=True)
batch_size = 64
#%%

# item_df['item_name'][i:i+batch_size].tolist()
#%%
for i in tqdm(range(0, len(item_df), batch_size)):
    # print(i)
    item_names = item_df['item_name'][i:i+batch_size]
    # 生成输出
    item_names = item_names.tolist()
    item_names = ['book: '+ item for item in item_names]
    inputs = tokenizer(item_names, return_tensors="pt", padding=True, truncation=True, max_length=128).to(model.device)
    with torch.no_grad():
        output = model(**inputs, output_hidden_states=True)
    seq_embeds = output.hidden_states[-1][:, -1, :].detach().cpu().numpy()
    # break
    if i == 0:
        item_llama_embeds = seq_embeds
    else:
        item_llama_embeds = np.concatenate((item_llama_embeds, seq_embeds), axis=0)
    # break

save_path = "item_info/"
np.save(op.join(save_path, "item_cf_embeds_llama3_7b_instruct_array_prefix.npy"), item_llama_embeds)

print("finished")