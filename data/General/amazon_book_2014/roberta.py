#%%
from transformers import RobertaTokenizer, RobertaModel
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
model_path = 'FacebookAI/roberta-base'
model = RobertaModel.from_pretrained(model_path, device_map = 'auto',token='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')
tokenizer = RobertaTokenizer.from_pretrained(model_path, device_map = 'auto',token='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')
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token
item_df = pd.read_csv('raw_data/items_filtered.csv', index_col=0)
item_df.rename(columns={'title': 'item_name'}, inplace=True)
batch_size = 64
#%%
batch_size = 128
for i in range(0, len(item_df), batch_size):
    print(i)
    item_names = item_df['item_name'][i:i+batch_size]
    inputs = tokenizer(item_names.tolist(), return_tensors="pt", padding=True, truncation=True)
    # break
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    if i == 0:
        item_bert_embeds = outputs.pooler_output.detach().numpy()
    else:
        item_bert_embeds = np.concatenate((item_bert_embeds, outputs.pooler_output.detach().numpy()), axis=0)
#%%
# len(outputs.hidden_states)
# # outputs.hidden_states[-1].
# outputs.pooler_output
# outputs.hidden_states[-1][:,0,:] == outputs.pooler_output
# outputs
item_bert_embeds.shape
#%%
save_path = "item_info/"
np.save(op.join(save_path, "item_cf_embeds_roberta_array.npy"), item_bert_embeds)

print("finished")
# %%
