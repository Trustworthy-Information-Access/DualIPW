import argparse

parser = argparse.ArgumentParser(description='Pipeline commandline argument')
parser.add_argument("--n_gpus", type=int, default=2,
                    help='The number of GPUs.')
parser.add_argument("--lr", type=float, default=2e-6,
                    help='The max learning rate for pre-training, and the learning rate for finetune.')
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--n_queries_for_each_gpu", type=int, default=5,
                    help='The number of training queries for each GPU. The size of training batch is based on this.')

parser.add_argument('--tra_type', type=str,
                    default='tradition', help='tradition total')
parser.add_argument('--data_path', type=str, default='./ntcir_data')
parser.add_argument('--gen_sim', type=int, default=1)
parser.add_argument('--tra_feat_size', type=int, default=13)
parser.add_argument("--combine", type=int, default=1,
                    help='Whether to use ltr_model')
parser.add_argument('--pro_size', type=int, default=14)
parser.add_argument('--total', type=int, default=0,
                    help='whether to add embedding together with score')
parser.add_argument('--label_type', type=str,
                    default='labels', help='labels iter_labels')
parser.add_argument("--num_candidates", type=int, default=10,
                    help="The number of candicating documents for each query in training data.")
parser.add_argument("--seed", type=int, default=0, help="seed")
parser.add_argument("--click_mode", type=str, default='multi',
                    help="way to generate train data with multi or single click")

parser.add_argument('--click_pos',type=int,default=-1)
parser.add_argument('--eval_type', type=str, default='validQsplit',
                    help='validQsplit devQsplit total_validQsplit')
parser.add_argument('--click_num',type=str,default='total',help='train data selection: total single multi')
parser.add_argument("--buffer_size", type=int, default=200000,
                    help='The size of training buffer. Depends on your memory size.')
parser.add_argument("--log_interval", type=int, default=10,
                    help='The number of interval steps to print logs.')
parser.add_argument("--eval_step", type=int, default=500,
                    help='The number of interval steps to validate.')
parser.add_argument("--save_step", type=int, default=5000,
                    help='The number of interval steps to save the model.')
parser.add_argument("--eval_batch_size", type=int,
                    default=30, help='The batchsize of evaluation.')
parser.add_argument("--output_name", type=str,
                    default='save_model', help="output_name.")
parser.add_argument("--epoch", type=int, default=1,
                    help="train epochs in task1")
parser.add_argument('--init_model', type=str, default='',
                    help='model loaded if iter==1')
parser.add_argument('--iter', type=int, default=0,
                    help='whether to choose PU learning iter step by step')

parser.add_argument("--method_name", type=str, default="",
                    help='The name of baseline. candidates: [IPWrank, DLA, RegressionEM, PairDebias, NavieAlgorithm]')
parser.add_argument('--ipw',type=int,default=0)
parser.add_argument('--limit',type=int,default=10)
parser.add_argument("--ranking_method", type=str, default="DNN",
                    help='The name of ranking_model. candidates: [DNN,...]')

parser.add_argument("--projection", type=int, default=0,
                    help="whether to project features")
parser.add_argument('--pro_bef', type=int, default=14, help='total feats')
parser.add_argument('--pro_aft', type=int, default=64,
                    help='64 when 14 feats')
parser.add_argument("--rank_feature_size", type=int,
                    default=64, help='The number of features used in ltr')

# for propensity_model in iobm
parser.add_argument('--s_nhidden',type=int,default=16)
parser.add_argument('--s_nlayers', type=int, default=1)
parser.add_argument('--click_dim', type=int, default=16)
parser.add_argument('--p_model', type=str, default='DenoisingNet')
parser.add_argument('--bi', type=int, default=1,
                    help='whether to model bi-direction')

# for qldenoise
parser.add_argument('--f_mode',type=str,default='kl')
parser.add_argument('--s1_nlayers',type=int,default=1)
parser.add_argument('--s1_hidden',type=int,default=8)

parser.add_argument('--loss_mode', type=str, default='listwise')

parser.add_argument('--eta',type=float,default=0)

config = parser.parse_args()

config.train_batch_size = config.n_gpus * config.n_queries_for_each_gpu * \
        config.num_candidates
config.exp_settings = {
    'method_name': config.method_name,
    'n_gpus': config.n_gpus,
    'lr': config.lr,
    'max_candidate_num': config.num_candidates,
    # same as candidate num not including true negatives
    'selection_bias_cutoff': config.num_candidates,
    'train_input_hparams': "",
    'learning_algorithm_hparams': "",
    'combine': config.combine,
    'rank_feature_size': config.rank_feature_size,  # unbiased
    'query_num': config.n_queries_for_each_gpu
}