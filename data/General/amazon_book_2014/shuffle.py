#%%
import numpy as np
import os.path as op
save_path = 'item_info'
# embeds_name = ""
embeds_name = "item_cf_embeds_large3_array.npy"
# data = np.load("/storage_fast/lhsheng/lhsheng/Tuning/RecRepo/data/General/amazon_movie/item_info/item_cf_embeds_large3_array.npy")
data = np.load(op.join(save_path, embeds_name))
data.shape
# %%
np.random.seed(0)
data.shape
data_shuffled = np.copy(data)
np.random.shuffle(data_shuffled)
#%%
data_shuffled.shape
embeds_name = "item_cf_embeds_large3_array_shuffle"
np.save(op.join(save_path, embeds_name), data_shuffled)
#%%
data_shuffled[1]
# import 
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
seq_target = cosine_similarity(data_shuffled,data)
ax = sns.heatmap(seq_target, cmap="RdBu",
                 square=True,  # 正方形格仔
                #  cbar=False,  # 去除 color bar
                #  xticklabels=False, yticklabels=False)
)
fig = ax.get_figure()
plt.show()
#%%
seq_target_2 = cosine_similarity(data,data)
#%%
ax = sns.heatmap(seq_target_2, cmap="RdBu",
                 square=True,  # 正方形格仔
                #  cbar=False,  # 去除 color bar
                #  xticklabels=False, yticklabels=False)
)
fig = ax.get_figure()
plt.show()