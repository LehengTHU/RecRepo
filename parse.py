import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # General Args
    parser.add_argument('--rs_type', type=str, default='LLM',
                        choices=['Seq', 'LLM', 'General'],
                        help='Seq, LLM, General')
    parser.add_argument('--model_name', type=str, default='SASRec',
                        help='model name.')
    parser.add_argument('--dataset', nargs='?', default='yc',
                        help='yc, ks, rr')

    parser.add_argument('--cuda', type=int, default=0,
                        help='cuda device.')
    parser.add_argument('--test_only', action="store_true",
                        help='Whether to test only.')
    parser.add_argument('--clear_checkpoints', action="store_true",
                        help='Whether clear the earlier checkpoints.')
    parser.add_argument('--saveID', type=str, default='Saved',
                        help='Specify model save path. Description of the experiment')
    
    parser.add_argument('--seed', type=int, default=101,
                        help='Random seed.')
    parser.add_argument('--max_epoch', type=int, default=500,
                        help='Number of max epochs.')
    parser.add_argument('--verbose', type=float, default=5,
                        help='Interval of evaluation.')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping point.')

    parser.add_argument("--mix", action="store_true",
                        help="whether to use mixed dataset")

    # Model Args
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size.')
    # parser.add_argument('--lr', type=float, default=0.001,
    #                     help='Learning rate.')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate.')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='weight decay for optimizer.')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout ')
    parser.add_argument("--no_wandb", action="store_true",
                        help="whether to use wandb")

    args, _ = parser.parse_known_args()

    if(args.rs_type == 'General'):
        parser.add_argument("--candidate", action="store_true",
                            help="whether using the candidate set")
        parser.add_argument('--Ks', type = int, default= 5,
                            help='Evaluate on Ks optimal items.')
        parser.add_argument('--neg_sample',type=int,default=1)
        parser.add_argument('--infonce', type=int, default=0,
                    help='whether to use infonce loss or not')
        parser.add_argument("--train_norm", action="store_true",
                            help="train_norm")
        parser.add_argument("--pred_norm", action="store_true",
                            help="pred_norm")
        parser.add_argument('--n_layers', type=int, default=0,
                            help='Number of GCN layers')
        parser.add_argument('--data_path', nargs='?', default='./data/General/',
                            help='Input data path.')
        parser.add_argument("--nodrop", action="store_true",
                            help="whether to drop out the enhanced training dataset")
        parser.add_argument('--num_workers', type=int, default=8,
                            help='number of workers in data loader')
        parser.add_argument('--regs', type=float, default=1e-5,
                            help='Regularization.')
        parser.add_argument('--max2keep', type=int, default=1,
                            help='max checkpoints to keep')
        args, _ = parser.parse_known_args()
        
        
        # INFONCE
        if(args.model_name == 'InfoNCE'):
            parser.add_argument('--tau', type=float, default=0.1,
                            help='temperature parameter')
        
        # MultVAE
        if(args.model_name == 'MultVAE'):
            parser.add_argument('--total_anneal_steps', type=int, default=200000,
                            help='total anneal steps')
            parser.add_argument('--anneal_cap', type=float, default=0.2,
                            help='anneal cap')
            parser.add_argument('--p_dim0', type=int, default=200,
                            help='p_dim0')
            parser.add_argument('--p_dim1', type=int, default=600,
                            help='p_dim1')

        # AgentRerank
        if('IntentCF' in args.model_name):
            parser.add_argument('--tau', type=float, default=0.1,
                            help='temperature parameter')

        if('LIntCF' in args.model_name):
            parser.add_argument('--tau', type=float, default=0.1,
                            help='temperature parameter')
            
            parser.add_argument('--lambda_cl', type=float, default=1,
                                    help='Rate of contrastive loss')
            parser.add_argument('--temp_cl', type=float, default=0.15,
                                    help='Temperature of contrastive loss')
            parser.add_argument('--eps', type=float, default=0.1,
                                help='Noise rate')

        #BC_LOSS
        if(args.model_name == 'BC_Loss'):
            parser.add_argument('--tau1', type=float, default=0.07,
                                help='temperature parameter for L1')
            parser.add_argument('--tau2', type=float, default=0.1,
                                help='temperature parameter for L2')
            parser.add_argument('--w_lambda', type=float, default=0.5,
                                help='weight for combining l1 and l2.')
            parser.add_argument('--freeze_epoch',type=int,default=5)

        #SimpleX
        if(args.model_name == 'SimpleX'):
            parser.add_argument('--w_neg', type=float, default=1)
            parser.add_argument('--neg_margin',type=float, default=0.4)
        
        # SGL
        if(args.model_name == 'SGL'):
            parser.add_argument('--lambda_cl', type=float, default=0.2,
                                help='Rate of contrastive loss')
            parser.add_argument('--temp_cl', type=float, default=0.15,
                                help='Temperature of contrastive loss')
            parser.add_argument('--droprate', type=float, default=0.1,
                            help='drop out rate for SGL')

        # XSimGCL
        if(args.model_name == "XSimGCL" or args.model_name == "RLMRec"):
            parser.add_argument('--lambda_cl', type=float, default=0.1,
                                help='Rate of contrastive loss')
            parser.add_argument('--temp_cl', type=float, default=0.15,
                                help='Temperature of contrastive loss')
            parser.add_argument('--layer_cl', type=int, default=1,
                                help='Number of layers for contrastive loss')
            parser.add_argument('--eps_XSimGCL', type=float, default=0.1,
                                help='Noise rate for XSimGCL')
        # RLMRec
        if(args.model_name == 'RLMRec'):
            parser.add_argument('--kd_temperature', type=float, default=0.2,
                            help='temperature parameter')
            parser.add_argument('--kd_weight', type=float, default=1e-2,
                            help='kd_weight')

    if(args.rs_type == 'Seq'):
        parser.add_argument('--r_click', type=float, default=0.2,
                            help='reward for the click behavior.')
        parser.add_argument('--r_buy', type=float, default=1.0,
                            help='reward for the purchase behavior.')
        parser.add_argument('--save_flag', type=int, default=0,
                            help='0: Disable model saver, 1: Activate model saver')
        parser.add_argument('--loss', type=str, default='bpr',
                            choices=['bpr', 'bce', 'mse'],
                            help='loss function.')
        args, _ = parser.parse_known_args()

        # Model-specific Args
        if(args.model_name == 'Caser'):
            parser.add_argument('--num_filters', type=int, default=16,
                                help='num_filters')
            parser.add_argument('--filter_sizes', nargs='?', default='[2,3,4]',
                                help='Specify the filter_size')

        if(args.model_name == 'GRU4Rec'):
            parser.add_argument('--gru_layers', type=int, default=1,
                                help='number of gru layers.')

        if(args.model_name == 'SASRec'):
            parser.add_argument('--num_heads', type=int, default=1,
                                help='num_heads')

    if(args.rs_type == 'LLM'):
        parser.add_argument('--llm_path', type=str, default='meta-llama/Llama-2-7b-hf',
                        help='path to llm model')
        parser.add_argument('--micro_batch_size', type=int, default=32,
                        help='micro batch size')
        
        args, _ = parser.parse_known_args()

        if(args.model_name == 'RecInt'):
            parser.add_argument('--recommender', type=str, default='SASRec',
                            help='SASRec, Caser, GRU4Rec, DreamRec')
            parser.add_argument('--rec_size', type=int, default=64,
                                help='embedding size of the recommender')
            parser.add_argument('--rec_type', type=str, default='h_all',
                            help='recommender type')
            parser.add_argument('--max_txt_len', type=int, default=32,
                                help='max text length')
            parser.add_argument('--end_sym', type=str, default='\n',
                                help='end symbol')
            # SASRec
            parser.add_argument('--num_heads', type=int, default=1,
                                help='num_heads')
            
        if(args.model_name == 'TALLRec'):
            parser.add_argument('--sample_num', type=int, default=64,
                                help='Sample number for training')
            parser.add_argument('--cutoff_len', type=int, default=512,
                                help='cutoff length')
            parser.add_argument('--not_train_on_inputs', action="store_true",
                                help='If True, masks out inputs in loss')
            parser.add_argument('--not_group_by_length', action="store_true",
                                help='If False, groups samples by length')
        # if(args.model_name == 'PerRec'):
        #     parser.add_argument('--llm_path', type=str, default='meta-llama/Llama-2-7b-hf',
        #                     help='path to llm model')

    # args_full, _ = parser.parse_known_args()

    # return args_full

    args_full, _ = parser.parse_known_args()
    special_args = list(set(vars(args_full).keys()) - set(vars(args).keys()))
    special_args.sort()

    return args_full, special_args
