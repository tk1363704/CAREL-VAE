import sys, os, warnings, time
import optuna
from drl_classifier_ec_mmd_final_mul_search import *
from bow_util import get_bow_zh

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--max_len', type=int, default=128, help='sentence max length')
parser.add_argument('--e_num_class', type=int, default=6, help='number of emotion class')
parser.add_argument('--c_num_class', type=int, default=1, help='number of cause class')
parser.add_argument('--pair_num_class', type=int, default=1, help='number of pair class')
parser.add_argument('--ec_dim', type=int, default=24, help='emotion or cause embedding dimension')
parser.add_argument('--bert_dim', type=int, default=768, help='bert embedding dimension')
parser.add_argument('--kl_ann_iterations', type=int, default=20000, help='kl annealing max iterations')
parser.add_argument('--epochs', type=int, default=20, help='training epochs')
parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
parser.add_argument('--ec_kl_lambda', type=float, default=0.03, help='emotion and cause kl weight')
parser.add_argument('--label_smoothing', type=float, default=0.1, help='label smoothing')
parser.add_argument('--mmd_loss_weight', type=float, default=30, help='emotion multitask loss weight')  # candidate: 30
parser.add_argument('--emo_mul_loss_weight', type=float, default=10, help='emotion multitask loss weight')
parser.add_argument('--cau_mul_loss_weight', type=float, default=10, help='cause multitask loss weight')
parser.add_argument('--pair_mul_loss_weight', type=float, default=30, help='pair multitask loss weight')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout probability')
parser.add_argument('--epsilon', type=float, default=1e-8, help='epsilon')
parser.add_argument('--vae_lr', type=float, default=7e-06, help='vae learning rate') # candidate: 1e-05
parser.add_argument('--bow_file', type=str, default='data/all_data_pair_zh.txt', help='bag of word file')
parser.add_argument('--best_model_path', type=str, default='ECPE_model/best_cause_pair_model',
                    help='best model saved path')
parser.add_argument('--self_iteration', type=int, default=50, help='self-training iteration')
parser.add_argument('--self_epochs', type=int, default=10, help='self-training epochs')
parser.add_argument('--self_strategy', type=str, default='random', help='self-training strategy')

opt = parser.parse_args()
opt.model_id = str(uuid4())
bow = get_bow_zh(opt.bow_file)
opt.pair_bow_dim = len(bow)

timestr = time.strftime("%Y%m%d-%H%M%S")
log = open('drl_search_log_' + timestr + '.txt', 'w', buffering=1)
sys.stdout = log
sys.stderr = log

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def objective(trial):

    params = {
              'mmd_loss_weight': trial.suggest_int("mmd_loss_weight", 10, 50, step=5),
              'emo_mul_loss_weight': trial.suggest_int("emo_mul_loss_weight", 10, 50, step=5),
              'cau_mul_loss_weight': trial.suggest_int("cau_mul_loss_weight", 10, 50, step=5),
              'pair_mul_loss_weight': trial.suggest_int("pair_mul_loss_weight", 10, 50, step=5),
              'vae_lr': trial.suggest_loguniform('vae_lr', 1e-6, 1e-5),
              'dropout': trial.suggest_float("dropout", 0.1, 0.9, step=0.1)
              }


    
    opt.mmd_loss_weight = params['mmd_loss_weight']
    opt.emo_mul_loss_weight = params['emo_mul_loss_weight']
    opt.cau_mul_loss_weight = params['cau_mul_loss_weight']
    opt.pair_mul_loss_weight = params['pair_mul_loss_weight']
    opt.vae_lr = params['vae_lr']
    opt.dropout = params['dropout']

    max_ec_f1 = train_overall(opt,device,trial)

    return max_ec_f1

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=100)

best_trial = study.best_trial

for key, value in best_trial.params.items():
    print("Best combination of hyperparameters:")
    print("{}: {}".format(key, value))