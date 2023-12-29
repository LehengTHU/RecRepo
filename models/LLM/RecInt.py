import os
import random
import time
import argparse
from tqdm import tqdm
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from collections import defaultdict
from termcolor import colored, cprint

from transformers import LlamaForCausalLM, LlamaTokenizer

# To Fix
from ..Seq.SASRec import SASRec
from ..Seq.Caser import Caser
from ..Seq.GRU4Rec import GRU4Rec
from ..Seq.SASRec_conv import SASRec_conv

from .base.data_utils import *
from .base.model_utils import *
from .base.utils import *
from .base.abstract_RS import AbstractRS
# from .base.abstract_model import AbstractModel

# from evaluator import *

class RecInt_RS(AbstractRS):
    def __init__(self, args) -> None:
        super().__init__(args)

    def add_additional_args(self):
        if(self.dataset_name == 'test_data'):
            self.args.state_size = 10
            self.args.item_num = 1682
    
    def preperation(self):
        super().preperation()
        if(os.path.exists(os.path.join(self.base_path, 'generation.txt'))):
            os.remove(os.path.join(self.base_path, 'generation.txt'))
        
    def load_model(self):
        exec('from models.LLM.'+ self.model_name + ' import ' + self.model_name)
        print('Model %s loaded!' % (self.model_name))
        self.model = eval(self.model_name + '(self.args)')

    def set_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, eps=1e-8, weight_decay=self.weight_decay)
        self.scheduler = lr_scheduler.LinearLR(self.optimizer, start_factor=0.2, end_factor=1, total_iters=5)

    def train_one_epoch(self, epoch):
        running_loss, num_batches = 0, 0
        len_train = len(self.data.train_loader)
        pbar = tqdm(enumerate(self.data.train_loader), mininterval=2, total = len_train)
        if self.verbose < 1:
            next_eval_point = self.verbose
            print("val interval is {}".format(int(next_eval_point*len_train)))
        for batch_i, batch in pbar:
            self.optimizer.zero_grad()
            loss = self.model.forward(batch)

            loss.backward()
            self.optimizer.step()
            running_loss += loss.detach().cpu().numpy()
            num_batches += 1

            if num_batches % 5000 == 0:
                self.scheduler.step()
            curr_epoch_progress = batch_i/len_train
            if self.verbose < 1 and curr_epoch_progress > next_eval_point: # evaluate the model
                self.save_log(f'Generation at epoch {epoch + curr_epoch_progress}', 'generation.txt')
                self.eval_and_check_early_stop(epoch, epoch_progress = round(curr_epoch_progress, 4))
                next_eval_point += self.verbose
                print(int(curr_epoch_progress*len_train), int(next_eval_point*len_train))
            
            # if(batch_i > 100):
            #     break
        return [running_loss/num_batches]

    @torch.no_grad()
    def evaluate(self, model, eval_data, device, name='valid'):
        test_dataloader = DataLoader(eval_data, batch_size=32, shuffle=False)

        n_total = 0
        n_recover = 0
        j = 0
        recover_dict = defaultdict(int)
        for _, sample in tqdm(enumerate(test_dataloader), mininterval=2, total=len(test_dataloader)):
            # print(_)
            output = self.model.generate_output(sample)  
        #     with torch.no_grad():
        #         output = self.model.generate_output(sample)
        
            j += 1

            for o, m in zip(output, sample["movie_seq"]):

                if j % 3 == 0:

                    self.save_log('Generated: ' + o, 'generation.txt')
                    mm = m.split('::')
                    mm = ', '.join(mm)
                    mm = 'This user has watched ' + mm + ' in the previous'
                    self.save_log('Real: ' + mm, 'generation.txt')
                    

                n_total += 1
                n_recover_this = 0
                real_movie_list = m.split('::')
                for movie in real_movie_list:
                    if re.search(movie, o):
                        n_recover += 1
                        n_recover_this += 1

                # print('recover {} movies'.format(n_recover_this))
                # print('**************************')
                
                recover_dict[n_recover_this] += 1

        for i in recover_dict.keys():
            recover_dict[i] = recover_dict[i] / n_total

        Ave_Recover = n_recover / n_total
        performance = {"Avg_Recover": Ave_Recover}
        perf_str = name+':{}'.format(performance)
        with open(self.base_path + '_stats.txt','a') as f:
            f.write(perf_str+"\n")
        print(performance)

        return Ave_Recover

    def save_log(self, one_line, file_name):
        full_file_name = os.path.join(self.base_path, file_name)
        with open(full_file_name, 'a') as f:
            f.write(one_line + '\n')

    def save_checkpoint(self, model, epoch, checkpoint_dir, epoch_progress=0):
        save_model = model.llama_proj
        state = {
            'epoch': epoch,
            'epoch_progress': epoch_progress,
            'state_dict': save_model.state_dict(),
        }
        cp_files = [file_ for file_ in os.listdir(checkpoint_dir)
                    if file_.startswith('epoch=') and file_.endswith('.checkpoint.pth.tar')]
        for cp in cp_files:
            os.remove(os.path.join(checkpoint_dir, cp))

        filename = os.path.join(checkpoint_dir, 'epoch={}.checkpoint.pth.tar'.format(epoch+epoch_progress))
        torch.save(state, filename)

    def restore_checkpoint(self, model, checkpoint_dir, device, force=False, pretrain=False):
        """
        If a checkpoint exists, restores the PyTorch model from the checkpoint.
        Returns the model and the current epoch.
        """
        cprint("[INFO] Start restoring the checkpoint", color='green', attrs=['bold'])
        cp_files = [file_ for file_ in os.listdir(checkpoint_dir)
                    if file_.startswith('epoch=') and file_.endswith('.checkpoint.pth.tar')]
        
        if(self.clear_checkpoints):
            for cp in cp_files:
                os.remove(os.path.join(checkpoint_dir, cp))
            cprint("[INFO] Checkpoints cleared", color='yellow', attrs=['bold'])
            return

        if(len(cp_files) == 0):
            cprint("[INFO] No checkpoint found", color='yellow', attrs=['bold'])
        else:
            filename = cp_files[0]
            full_filepath = os.path.join(checkpoint_dir, filename)
            checkpoint = torch.load(full_filepath, map_location = str(device))
            self.model.llama_proj.load_state_dict(checkpoint['state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            epoch_progress = checkpoint['epoch_progress']
            cprint(f"[INFO] Checkpoint at epoch {self.start_epoch+epoch_progress} loaded", color='yellow', attrs=['bold'])

class RecInt(nn.Module):
    # def __init__(self, hidden_size, item_num, state_size, num_filters, filter_sizes,
    #              dropout_rate):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.dataset_name = args.dataset
        self.rec_size = args.rec_size
        self.recommender = args.recommender
        self.rec_type = args.rec_type
        # if args.llm_path == 'LLAMA2':
            # self.llm_path = "meta-llama/Llama-2-7b-hf"
        self.llm_path = args.llm_path
        self.max_txt_len = args.max_txt_len
        self.end_sym = args.end_sym

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 读取推荐模型
        print('Loading Rec Model')
        print('./assets/recint/pretrained/{}_{}_{}.pt'.format(self.recommender, self.dataset_name, self.rec_type))
        # self.rec_model = torch.load('./assets/recint/pretrained/{}_{}_{}.pt'.format(self.recommender, self.dataset_name, self.rec_type))

        # self.rec_model = SASRec(args)
        self.rec_model = SASRec_conv(args)
        self.rec_model.load_state_dict(torch.load('./assets/recint/pretrained/{}_{}_{}.pt'.format(self.recommender, self.dataset_name, self.rec_type)))

        self.rec_model.eval()
        # 固定参数
        for name, param in self.rec_model.named_parameters():
            param.requires_grad = False
        self.rec_model.to(device)
        print('device', self.rec_model.device)
        print(self.rec_model.item_embeddings.weight.device)

        print('Loding Rec model {} Done'.format(self.recommender))

        # 读取数据和prompt
        if self.dataset_name == 'ml100k' or self.dataset_name == 'game' or self.dataset_name == 'game2' or self.dataset_name == 'steam' or self.dataset_name == 'test_data':
            prompt_path="./assets/recint/prompt/{}/alignment_seq.txt".format(self.dataset_name)
            prompt_path_test="./assets/recint/prompt/{}/alignment_seq_test.txt".format(self.dataset_name)
        else:
            raise ValueError("no dataset: {}".format(self.dataset_name))
        
        prompt_template="{}"
        # 处理prompt
        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<SeqHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

        if prompt_path_test:
            with open(prompt_path_test, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<SeqHere>" in raw_prompt]
            self.prompt_list_test = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} test prompts'.format(len(self.prompt_list_test)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list_test)))
        else:
            self.prompt_list = []

        print('Loading LLAMA')

        # 读取huggingface模型
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(self.llm_path, use_fast=False)
        self.llama_tokenizer.pad_token = "$$"
        self.llama_tokenizer.add_special_tokens({'additional_special_tokens': ['<Seq>', '</Seq>']})
        # self.llama_tokenizer.mol_token_id = self.llama_tokenizer("<mol>", add_special_tokens=False).input_ids[0]

        # self.llama_model = LlamaForCausalLM.from_pretrained(llama_model,load_in_8bit=True,torch_dtype=torch.float16, device_map="auto")
        self.llama_model = LlamaForCausalLM.from_pretrained(self.llm_path,device_map="auto")
        # self.llama_model = LlamaForCausalLM.from_pretrained(llama_model,load_in_8bit=True,device_map="auto")
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False

        print('Loading LLAMA Done')

        # 读取linear层
        self.llama_proj = nn.Linear(
            self.rec_size, self.llama_model.config.hidden_size
        )
        self.llama_proj.to(device)

        self.act_f = nn.ReLU()
        # self.max_txt_len = max_txt_len
        # self.end_sym = end_sym

    def encode_seq(self, seq, len_seq):

        device = seq.device
        seq_emb = self.rec_model.cacul_h(seq, len_seq) #should be (B, 1, rec_dim)
        # seq_emb = self.rec_model.cacu_h(seq, len_seq) #should be (B, 1, rec_dim)

        if self.recommender == 'Caser' or self.recommender == 'Dream' or self.recommender == 'GRU':
            seq_emb = seq_emb.view(seq_emb.shape[0], 1, seq_emb.shape[1])

        # print(seq_emb.shape)
        seq_input_llama = self.llama_proj(seq_emb)
        seq_atts_llama = torch.ones(seq_input_llama.size()[:-1], dtype=torch.long).to(seq.device)

        # seq_input_llama = self.act_f(seq_input_llama)
        return seq_input_llama, seq_atts_llama
    
    def encode_seq_gru(self, seq, len_seq):

        device = seq.device

        seq_emb = self.rec_model.cacul_h(seq, len_seq) #should be (B, 1, rec_dim)
        if self.recommender == 'Caser' or self.recommender == 'Dream' or self.recommender == 'GRU':
            seq_emb = seq_emb.view(seq_emb.shape[0], 1, seq_emb.shape[1])

        # seq_emb = self.rec_model.cacu_h(seq, len_seq) #should be (B, 1, rec_dim)

        # if self.recommender == 'Caser' or self.recommender == 'Dream':
        #     seq_emb = seq_emb.view(seq_emb.shape[0], 1, seq_emb.shape[1])

        # print(seq_emb.shape)
        seq_input_llama = self.llama_proj(seq_emb)
        seq_atts_llama = torch.ones(seq_input_llama.size()[:-1], dtype=torch.long).to(seq.device)

        # seq_input_llama = self.act_f(seq_input_llama)
        return seq_input_llama, seq_atts_llama

    def get_context_emb(self, prompt, seq_list):
        device = seq_list[0].device

        prompt_segs = prompt.split('<SeqHere>')

        seg_tokens = [
            self.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]

        seg_embs = [self.embed_tokens(seg_t) for seg_t in seg_tokens]

        mixed_embs = [emb for pair in zip(seg_embs[:-1], seq_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)

        return mixed_embs


    def prompt_wrap(self, seq_embeds, atts_seq, prompts):
        if prompts:
            emb_lists = []
            if isinstance(prompts, str):
                prompts = [prompts] * len(seq_embeds)

            for each_seq_embed, each_prompt in zip(seq_embeds, prompts):
                p_before, p_after = each_prompt.split('<SeqHere>')

                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False).to(seq_embeds.device)
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False).to(seq_embeds.device)
                p_before_embed = self.embed_tokens(p_before_tokens.input_ids)
                p_after_embed = self.embed_tokens(p_after_tokens.input_ids)
                wrapped_emb = torch.cat([p_before_embed, each_seq_embed[None], p_after_embed], dim=1)
                emb_lists.append(wrapped_emb)
            emb_lens = [emb.shape[1] for emb in emb_lists]
            pad_emb = self.embed_tokens(torch.tensor(self.llama_tokenizer.pad_token_id, device=seq_embeds.device))
            wrapped_embs = pad_emb.expand(len(emb_lens), max(emb_lens), -1).clone()
            wrapped_atts = torch.zeros([len(emb_lens), max(emb_lens)], dtype=torch.int, device=seq_embeds.device)
            for i, emb in enumerate(emb_lists):
                wrapped_embs[i, :emb_lens[i]] = emb
                wrapped_atts[i, :emb_lens[i]] = 1
            return wrapped_embs, wrapped_atts

    def concat_emb_input_output(self, input_embs, input_atts, output_embs, output_atts):
        input_lens = []
        cat_embs = []
        cat_atts = []
        for i in range(input_embs.size(0)):
            input_len = input_atts[i].sum()
            input_lens.append(input_len)
            cat_embs.append(
                torch.cat([
                    input_embs[i][:input_len],
                    output_embs[i],
                    input_embs[i][input_len:]
                ])
            )
            cat_atts.append(
                torch.cat([
                    input_atts[i][:input_len],
                    output_atts[i],
                    input_atts[i][input_len:]
                ])
            )
        cat_embs = torch.stack(cat_embs)
        cat_atts = torch.stack(cat_atts)
        return cat_embs, cat_atts, input_lens

    def embed_tokens(self, token_ids):

        embeds = self.llama_model.base_model.embed_tokens(token_ids)
        return embeds


    def forward(self, samples):
        seq = samples["seq"]
        len_seq = samples['len_seq']
        len_seq_list = len_seq.tolist()
        if self.recommender == 'SASRec' or 'Caser':
            seq_embeds, atts_seq = self.encode_seq(seq, len_seq)
        elif self.recommender == 'GRU':
            seq_embeds, atts_seq = self.encode_seq_gru(seq, len_seq_list)

        if self.prompt_list:
            instruction = random.choice(self.prompt_list)
        else:
            instruction = samples["instruction_input"] if "instruction_input" in samples else None

        seq_embeds, atts_seq = self.prompt_wrap(seq_embeds, atts_seq, instruction)

        self.llama_tokenizer.padding_side = "right"

        movie_seq_list = [t.split('::') for t in samples["movie_seq"]]
        if self.dataset_name == 'ml100k' or self.dataset_name == 'test_data':
            movie_seq_str = [', '.join(t) for t in movie_seq_list]
            movie_seq_str = ['This user has watched ' + t + ' in the previous.' for t in movie_seq_str]
        elif self.dataset_name == 'game' or self.dataset_name == 'game2' or self.dataset_name == 'steam':
            movie_seq_str = [', '.join(t) for t in movie_seq_list]
            movie_seq_str = ['This user has played ' + t + ' in the previous.' for t in movie_seq_str]
        else:
            raise ValueError("no dataset: {}".format(self.dataset_name))

        text = [t + self.end_sym for t in movie_seq_str]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(seq.device)

        batch_size = seq_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                        dtype=to_regress_tokens.input_ids.dtype,
                        device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_seq[:, :1]

        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds, attention_mask, input_lens = \
            self.concat_emb_input_output(seq_embeds, atts_seq, to_regress_embeds, to_regress_tokens.attention_mask)
        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, attention_mask], dim=1)

        part_targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        targets = (
            torch.ones([inputs_embeds.shape[0], inputs_embeds.shape[1]],
                    dtype=torch.long).to(seq.device).fill_(-100)
        )

        for i, target in enumerate(part_targets):
            targets[i, input_lens[i] + 1:input_lens[i] + len(target) + 1] = target  # plus 1 for bos

        # with self.maybe_autocast():
        #     outputs = self.llama_model(
        #         inputs_embeds=inputs_embeds,
        #         attention_mask=attention_mask,
        #         return_dict=True,
        #         labels=targets,
        #     )
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        return loss
    

    @torch.no_grad()
    def generate_output(self, samples):
        seq = samples["seq"]
        len_seq = samples['len_seq']
        len_seq_list = len_seq.tolist()
        if self.recommender == 'SASRec' or 'Caser':
            seq_embeds, atts_seq = self.encode_seq(seq, len_seq)
        elif self.recommender == 'GRU':
            seq_embeds, atts_seq = self.encode_seq_gru(seq, len_seq_list)

        if self.prompt_list_test:
            instruction = random.choice(self.prompt_list_test)
        else:
            instruction = samples["instruction_input"] if "instruction_input" in samples else None

        instruction_tokens = self.llama_tokenizer(
            instruction,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(seq_embeds.device)

        seq_embeds, atts_seq = self.prompt_wrap(seq_embeds, atts_seq, instruction)

        batch_size = seq_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                        dtype=instruction_tokens.input_ids.dtype,
                        device=instruction_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_seq[:, :1]
        # atts_bos = torch.ones([batch_size, 1], dtype=torch.int, device=item_embeds.device),

        inputs_embeds = torch.cat([bos_embeds, seq_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_seq], dim=1)


        generate_ids = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_length=100
        )

        response = self.llama_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)  

        return response

