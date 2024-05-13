#%%
import numpy as np
import os
import os.path as op
# model_path = "jspringer/echo-mistral-7b-instruct-lasttoken"

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import os
import os.path as op
from tqdm import tqdm

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModel

print(os.environ["HF_HOME"])
# print(os.environ["HF_HUB_CACHE"])


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

model_path = "mistralai/Mistral-7B-v0.1"


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, device_map = 'auto')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
for param in model.parameters():
    param.requires_grad = False
# tokenizer.padding_side = "left"
# tokenizer.pad_token = tokenizer.eos_token


item_df = pd.read_csv('raw_data/items_filtered.csv', index_col=0)
item_df.rename(columns={'title': 'item_name'}, inplace=True)
item_df

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:# 向左边填充了所有的值。
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
batch_size = 64
max_length = 128

for i in tqdm(range(0,len(item_df),batch_size)):
    item_names = item_df['item_name'][i:i+batch_size]
    # 生成输出
    batch_dict = tokenizer(item_names.tolist(), max_length=max_length-1, padding=True, truncation=True, return_tensors="pt").to(model.device)
    input_ids_shape = batch_dict['input_ids'].shape
    input_ids_eos = torch.full((input_ids_shape[0], 1), tokenizer.eos_token_id, dtype=batch_dict.input_ids.dtype, device=batch_dict.input_ids.device)
    batch_dict.input_ids = torch.cat([batch_dict.input_ids, input_ids_eos], dim=1)
    EOS_mask_shape = batch_dict['attention_mask'].shape
    EOS_mask = torch.full((EOS_mask_shape[0], 1), 1, dtype=batch_dict.attention_mask.dtype, device=batch_dict.attention_mask.device)
    batch_dict.attention_mask = torch.cat([batch_dict.attention_mask, EOS_mask], dim=1)
    
    with torch.no_grad():
        outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = embeddings.detach().cpu().numpy()
    if i == 0:
        item_embeds = embeddings
    else:
        item_embeds = np.concatenate((item_embeds, embeddings), axis=0)

save_path = 'item_info'
embeds_name = model_path.split('/')[-1]
embeds_name = "item_cf_embeds_" + embeds_name +'_EOS' + "_array.npy"

embeds_name, item_embeds.shape, op.join(save_path, embeds_name)


np.save(op.join(save_path, embeds_name),item_embeds)


save_path = 'item_info'
embeds_name = model_path.split('/')[-1]
embeds_name = "item_cf_embeds_" + embeds_name +'_EOS' + "_array.npy"
data = np.load(op.join(save_path, embeds_name))
norms = np.linalg.norm(data, ord=2, axis=1, keepdims=True)
normalized_data = data / norms
normalized_data
squared_sums = np.sum(normalized_data**2, axis=1)
squared_sums

embeds_name = model_path.split('/')[-1]
embeds_name = "item_cf_embeds_" + embeds_name +'_EOS' + "_array.npy"
np.save(op.join(save_path, embeds_name), normalized_data)