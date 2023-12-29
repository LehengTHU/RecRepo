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
    parser.add_argument('--verbose', type=float, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping point.')

    # Model Args
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='weight decay for optimizer.')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout ')
    
    args, _ = parser.parse_known_args()

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


    args_full, _ = parser.parse_known_args()

    return args_full


