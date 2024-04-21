import pandas as pd
import torch
import numpy as np
import os
import os.path as op
from tqdm import tqdm


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

print(os.environ["HF_HOME"])
print(os.environ["HF_HUB_CACHE"])


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"



# model_path = "meta-llama/Meta-Llama-3-8B"
model_path = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_path, device_map = 'auto')
model = AutoModelForCausalLM.from_pretrained(model_path, device_map = 'auto')

tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token


item_df = pd.read_csv('raw_data/items_filtered.csv', index_col=0)
item_df.rename(columns={'title': 'item_name'}, inplace=True)

batch_size = 64

for i in tqdm(range(0, len(item_df), batch_size)):
    # print(i)
    item_names = item_df['item_name'][i:i+batch_size]
    # 生成输出
    inputs = tokenizer(item_names.tolist(), return_tensors="pt", padding=True, truncation=True, max_length=128).to(model.device)
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
np.save(op.join(save_path, "item_cf_embeds_llama2_7b_array.npy"), item_llama_embeds)

print("finished")