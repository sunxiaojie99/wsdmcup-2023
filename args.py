# -*- encoding: utf-8 -*-
'''
@Time    :   2022/06/14 15:33:57
@Author  :   Chu Xioakai
@Contact :   xiaokaichu@gmail.com
'''
import argparse

parser = argparse.ArgumentParser(description='Pipeline commandline argument')

# parameters for dataset settings
parser.add_argument("--train_datadir", type=str, default='./data/train_data/',
                    help="The directory of the training dataset.")
parser.add_argument("--valid_annotate_path", type=str, default='./data/annotate_data/val_data.txt',
                    help="The path of the valid/test annotated data.")
parser.add_argument("--valid_click_path", type=str, default='./data/click_data/part-00000.gz',
                    help="The path of the valid/test click data.")
parser.add_argument("--num_candidates", type=int, default=30,
                    help="The number of candicating documents for each query in training data.")
parser.add_argument("--ntokens", type=int, default=22000,
                    help="The number of tokens in dictionary.")
parser.add_argument("--seed", type=int, default=0, help="seed")

# parameters for Transformer
parser.add_argument("--max_seq_len", type=int, default=128,
                    help="The max sequence of input for Transformer.")
parser.add_argument("--emb_dim", type=int, default=128,
                    help="The embedding dim.")
parser.add_argument("--nlayers", type=int, default=2,
                    help="The number of Transformer encoder layer.")
parser.add_argument("--nhead", type=int, default=2,
                    help="The number of attention head.")
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--n_queries_for_each_gpu", type=int, default=5,
                    help='The number of training queries for each GPU. The size of training batch is based on this.')
parser.add_argument("--init_parameters", type=str, default='',
                    help='The warmup model for Transformer.')
parser.add_argument("--eval_batch_size", type=int,
                    default=2000, help='The batchsize of evaluation.')

# parameters for training
parser.add_argument("--n_gpus", type=int, default=2,
                    help='The number of GPUs.')
parser.add_argument("--lr", type=float, default=2e-6,
                    help='The max learning rate for pre-training, and the learning rate for finetune.')
parser.add_argument("--max_steps", type=int, default=100000,
                    help='The max number of training steps.')
parser.add_argument("--warmup_steps", type=int, default=4000)
parser.add_argument("--weight_decay", type=float, default=0.01)

# parameter for logs & save model
parser.add_argument("--buffer_size", type=int, default=500000,
                    help='The size of training buffer. Depends on your memory size.')
parser.add_argument("--log_interval", type=int, default=10,
                    help='The number of interval steps to print logs.')
parser.add_argument("--eval_step", type=int, default=500,
                    help='The number of interval steps to validate.')
parser.add_argument("--save_step", type=int, default=5000,
                    help='The number of interval steps to save the model.')

# parameter for baseline models in finetune
parser.add_argument("--method_name", type=str, default="NavieAlgorithm",
                    help='The name of baseline. candidates: [IPWrank, DLA, RegressionEM, PairDebias, NavieAlgorithm]')

# for submission
parser.add_argument("--test_annotate_path", type=str,
                    default='./data/wsdm_round_1/wsdm_test_1.txt', help="The path of the test annotated data.")
parser.add_argument("--evaluate_model_path", type=str,
                    default='', help="The path of saved model.")
parser.add_argument("--result_path", type=str, default='result.csv',
                    help="The path of saving submitting results.")
parser.add_argument("--finetune_epoch", type=int,
                    default=20, help="finetune epoch")

# for finetune
parser.add_argument("--negative_num", type=int, default=0,
                    help='Negative document num for a given positive doc, e.g. 1,2,...M.')
parser.add_argument("--pos_num", type=int, default=0,
                    help='')
parser.add_argument("--strategy", type=str, default='q',
                    help="Construct epoch dataset based on query, i.e. sample one pair (if negative num is 1) for each query (q); Or based on positive docs(d).")
parser.add_argument("--finetune_type", type=str,
                    default='finetune_point', help="finetune_pair finetune_point.")
parser.add_argument("--output_name", type=str,
                    default='save_model', help="output_name.")
parser.add_argument("--tem", type=float, default=1.0)

# for task1
parser.add_argument("--ranking_method", type=str, default="DNN",
                    help='The name of ranking_model. candidates: [DNN,...]')
parser.add_argument("--rank_feature_size", type=int,
                    default=792, help='The number of features used in ltr')
parser.add_argument("--projection", type=int, default=0,
                    help="whether to project features")
parser.add_argument("--combine", type=int, default=0,
                    help='Whether to use ltr_model')
parser.add_argument("--change_label", type=str, default='no',
                    help='Whether to change_label')
parser.add_argument("--train_load_feature", type=int, default=0,
                    help="whether to load features when training")
parser.add_argument("--feature_pairs", type=int,
                    default=[], help="number of pairs based on feature rank", action="append")
parser.add_argument("--mode", type=str, default='pair',
                    help="way to generate train data")
parser.add_argument("--pairs", type=int, default=5,
                    help="when mode is pair,number of pairs")
parser.add_argument("--task1_epoch", type=int, default=1,
                    help="train epochs in task1")
parser.add_argument("--vote", type=int, default=0,
                    help="whether to use vote_feature")


# task1 evaluation
parser.add_argument("--train_feature_path", type=str,
                    default='./data/annotate_data/train_features.txt', help="The path of the train features data.")
parser.add_argument("--valid_data_path", type=str, default='./data/annotate_data/valid_data.txt',
                    help="The path of the valid annotated data.")
parser.add_argument("--valid_feature_path", type=str,
                    default='./data/annotate_data/valid_features.txt', help="The path of the valid features data.")
parser.add_argument("--test_data_path", type=str, default='./data/annotate_data/test_data.txt',
                    help="The path of the test annotated data.")
parser.add_argument("--test_feature_path", type=str,
                    default='./data/annotate_data/test_features.txt', help="The path of the test features data.")
# task1 submission
parser.add_argument("--data_path", type=str, default='./data/test_data/wsdm_test_2_all.txt',
                    help="The path of the test annotated data.")
parser.add_argument("--feature_path", type=str, default='./data/test_data/test_2_features.txt',
                    help="The path of the test feature data.")

# choose feature to refine target
parser.add_argument("--feature_type", type=str, default="JM_title",
                    help="The feature chose to refine target")
parser.add_argument("--mean_std_file", type=str, default='./data/mean_std_file.txt',
                    help="initial data_files feature mean_std value")
parser.add_argument("--temperature", type=float, default=0.5,
                    help="tao used in softmax")
parser.add_argument("--delta", type=float, default=0.1,
                    help="delta used in process_target")

parser.add_argument("--freq_path", type=str, default='./data/query_frequency.txt',
                    help="the number of the query in total files")
parser.add_argument("--add_freq", type=str, default="False",
                    help='Whether to use freq change')
parser.add_argument("--freq_k", type=float, default=1,
                    help='hypoparameter for frequency')
parser.add_argument("--freq_b", type=float, default=1,
                    help='hypoparameter for frequency')

# for pretrain
parser.add_argument("--pretrain_epoch", type=int, default=1)

config = parser.parse_args()

config._CLS_ = 0
config._SEP_ = 1
config._PAD_ = 2
config._MASK_ = 3

candidate = config.num_candidates+config.negative_num
if config.mode == 'pair':
    candidate = config.pairs*2
    config.train_batch_size = 300
else:
    """ The size of training batch should be 'ngpus * nqueriy * n_candidates' """
    config.train_batch_size = config.n_gpus * config.n_queries_for_each_gpu * \
        candidate

""" The size of test batch is flexible. It depends on your memory. """

# The input-dict for baseline model.
config.exp_settings = {
    'method_name': config.method_name,
    'n_gpus': config.n_gpus,
    'init_parameters': config.init_parameters,
    'lr': config.lr,
    'max_candidate_num': config.num_candidates,
    'selection_bias_cutoff': config.num_candidates,  # same as candidate num
    'feature_size': config.emb_dim,
    'train_input_hparams': "",
    'learning_algorithm_hparams': "",
    'combine': config.combine,
    'rank_feature_size': config.rank_feature_size,  # unbiased
    'negative_num': config.negative_num,
    'change_label': config.change_label,
    'add_freq': config.add_freq,  
    'freq_k': config.freq_k,  
    'freq_b': config.freq_b,  
}
