"""
Written by Yujin Huang(Jinx)
Started 30/11/2021 11:11 pm
Last Editted 

Description of the purpose of the code
"""
import argparse
import torch
import torch.nn
from torch import nn
import time
import os, sys
from data_process import *
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from uuid import uuid4

#### setting agrparse
parser = argparse.ArgumentParser(description='Training')
##### input
parser.add_argument('--source_domain', default='society', help='society')
parser.add_argument('--target_domain', default='finance', help='finance, education, entertainment and home')
parser.add_argument('--max_sen_len', type=int, default=60, help='max number of tokens per sentence')
parser.add_argument('--max_doc_len', type=int, default=75, help='max number of sentences per document')
##### model struct
parser.add_argument('--n_hidden', type=int, default=100, help='number of hidden unit')
parser.add_argument('--n_class', type=int, default=7, help='number of distinct class')
##### traing
parser.add_argument('--training_iter', type=int, default=10, help='number of train iteration')
parser.add_argument('--self_iter', type=int, default=5, help='number of self train iteration')
parser.add_argument('--top_k', type=int, default=2, help='number of negative sentence in self training')
parser.add_argument('--threshold', type=int, default=0.7, help='confident threshold')
parser.add_argument('--batch_size', type=int, default=4, help='number of example per batch')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='learning rate')
parser.add_argument('--keep_softmax', type=float, default=1.0, help='softmax layer dropout keep probability')
parser.add_argument('--l2_reg', type=float, default=1e-5, help='l2 regularization')
parser.add_argument('--emotion', type=float, default=1.00, help='lambda')

torch.set_printoptions(profile="full")
opt = parser.parse_args()
timestr = time.strftime("%Y%m%d-%H%M%S")
log = open('A_log_' + timestr + '.txt', 'w', buffering=1)
sys.stdout = log
sys.stderr = log


def print_training_info():
    print('\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>TRAINING INFO:\n')
    print('batch-{}, lr-{}, kb_softmax-{}, l2_reg-{}'.format(
        opt.batch_size, opt.learning_rate, opt.keep_softmax, opt.l2_reg))
    # print('training_iter-{} self-training_iter--{} negative_top_k--{}'.format(opt.training_iter, opt.self_iter,
    #                                                                           opt.top_k))
    print('training_iter-{} confident_threshold-{}'.format(opt.training_iter, opt.threshold))
    print('source_domain-{}, target_domain--{}\n'.format(opt.source_domain, opt.target_domain))


def save_best_ckp(state, best_path, model_id):
    if not os.path.exists(best_path):
        os.makedirs(best_path)

    best_path = os.path.join(best_path, model_id + ".pt")
    torch.save(state, best_path)


def load_ckp(checkpoint_path, model):
    model.load_state_dict(torch.load(checkpoint_path))

    return model


def generate_pair_data(file_name, doc_id, doc_len, y_pairs, pred_y_cause, pred_y_emotion, x):
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext',
                                              cache_dir="model/hfl_chinese-roberta-wwm-ext")
    g = open(file_name, 'w')
    for i in range(len(doc_id)):
        g.write(doc_id[i] + ' ' + str(doc_len[i]) + '\n')
        g.write(str(', '.join(y_pairs[i])) + '\n')
        for j in range(doc_len[i]):
            clause = tokenizer.decode(x[i][j], skip_special_tokens=True)
            g.write(str(j + 1) + ', ' + str(pred_y_emotion[i][j]) + ', ' + str(
                pred_y_cause[i][j]) + ', ' + clause + '\n')
    print('write {} done'.format(file_name))


# def generate_self_train_data(target_data, best_pred, K):
#     self_y_emotion = np.zeros((best_pred.shape[0], best_pred.shape[1], best_pred.shape[2]))
#     for i in range(len(target_data.doc_id)):
#         emo_prob = {}
#         none_prob = {}
#         for j in range(target_data.doc_len[i]):
#             emotion = np.argmax(best_pred[i][j])
#             if emotion == 6:
#                 none_prob[str(j) + "," + str(emotion)] = best_pred[i][j][emotion]
#             else:
#                 emo_prob[str(j) + "," + str(emotion)] = best_pred[i][j][emotion]

#         # obtain the emotion sentence that has the highest score
#         sort_emo_prob = sorted(emo_prob.items(), key=lambda x: x[1], reverse=True)

#         # obtain top-k non-emotion sentence
#         sort_none_prob = sorted(none_prob.items(), key=lambda x: x[1], reverse=True)
#         top_k_none_prob = sort_none_prob[:K]

#         if sort_emo_prob and len(sort_emo_prob)>=2:
#             emo_sen_index = int(sort_emo_prob[0][0].split(",")[0])
#             emo_index = int(sort_emo_prob[0][0].split(",")[1])

#             emo_sen_index_two = int(sort_emo_prob[1][0].split(",")[0])
#             emo_index_two = int(sort_emo_prob[1][0].split(",")[1])
#             # emo_score = sort_emo_prob[0][1]
#             # positive (emotion) sentence
#             for j in range(target_data.doc_len[i]):
#                 if j == emo_sen_index:
#                     self_y_emotion[i][j][emo_index] = 1
#                 if j == emo_sen_index_two:
#                     self_y_emotion[i][j][emo_index_two] = 1

#             # negative (non-emotion) sentence
#             for j in range(len(top_k_none_prob)):
#                 non_emo_sen_index = int(top_k_none_prob[j][0].split(",")[0])
#                 self_y_emotion[i][non_emo_sen_index][-1] = 1
#         elif sort_emo_prob:
#             emo_sen_index = int(sort_emo_prob[0][0].split(",")[0])
#             emo_index = int(sort_emo_prob[0][0].split(",")[1])

#             for j in range(target_data.doc_len[i]):
#                 if j == emo_sen_index:
#                     self_y_emotion[i][j][emo_index] = 1

#             for j in range(len(top_k_none_prob)):
#                 non_emo_sen_index = int(top_k_none_prob[j][0].split(",")[0])
#                 self_y_emotion[i][non_emo_sen_index][-1] = 1
#         else:
#             for j in range(len(top_k_none_prob)):
#                 non_emo_sen_index = int(top_k_none_prob[j][0].split(",")[0])
#                 self_y_emotion[i][non_emo_sen_index][-1] = 1

#     target_data.y_emotion = self_y_emotion
#     self_train_loader = DataLoader(target_data, batch_size=opt.batch_size, shuffle=True)

#     return self_train_loader

def generate_self_train_data(source_data, best_pred):
    empty = False
    self_y_emotion = np.zeros((best_pred.shape[0], best_pred.shape[1], best_pred.shape[2]))
    self_train = "domains/THUCTC_multiple/" + opt.target_domain + ".txt"
    target_data = ECPE_Dataset(self_train, max_sen_len=opt.max_sen_len, max_doc_len=opt.max_doc_len)
    data_size = target_data.x_ids.shape[0]
    no_conf_list = []

    for i in range(len(target_data.doc_id)):
        emo_prob = {}
        none_prob = {}
        for j in range(target_data.doc_len[i]):
            emotion = np.argmax(best_pred[i][j])
            if emotion == 6:
                none_prob[str(j) + "," + str(emotion)] = best_pred[i][j][emotion]
            else:
                emo_prob[str(j) + "," + str(emotion)] = best_pred[i][j][emotion]

        # obtain the emotion sentence that has the highest score
        sort_emo_prob = sorted(emo_prob.items(), key=lambda x: x[1], reverse=True)
      

        # obtain top-k non-emotion sentence
        # sort_none_prob = sorted(none_prob.items(), key=lambda x: x[1], reverse=True)
        # top_k_none_prob = sort_none_prob[:K]

        if sort_emo_prob:
            emo_sen_index = int(sort_emo_prob[0][0].split(",")[0])
            emo_index = int(sort_emo_prob[0][0].split(",")[1])
            emo_prob = sort_emo_prob[0][1]
            # positive (emotion) sentence
            if emo_prob > opt.threshold:
                for j in range(target_data.doc_len[i]):
                    if j == emo_sen_index:
                        self_y_emotion[i][j][emo_index] = 1
                    else:
                        self_y_emotion[i][j][-1] = 1
            else:
                no_conf_list.append(i)

            # # negative (non-emotion) sentence
            # for j in range(len(top_k_none_prob)):
            #     non_emo_sen_index = int(top_k_none_prob[j][0].split(",")[0])
            #     self_y_emotion[i][non_emo_sen_index][-1] = 1
        else:
            no_conf_list.append(i)

    # x_ids, x_masks, x_types, doc_len, y_emotion, y_cause
    target_data.x_ids = np.delete(target_data.x_ids, no_conf_list, 0)
    target_data.x_masks = np.delete(target_data.x_masks, no_conf_list, 0)
    target_data.x_types = np.delete(target_data.x_types, no_conf_list, 0)
    target_data.doc_len = np.delete(target_data.doc_len, no_conf_list, 0)
    target_data.y_emotion = np.delete(self_y_emotion, no_conf_list, 0)
    target_data.y_cause = np.delete(target_data.y_cause, no_conf_list, 0)

    if target_data.x_ids.shape[0] < data_size:
        print("# total doc:", data_size)
        print("# confident doc:", target_data.x_ids.shape[0], '\n')
    else:
        empty = True

    conc_data = ConcatDataset([source_data, target_data])
    self_train_loader = DataLoader(conc_data, batch_size=opt.batch_size, shuffle=True)

    return self_train_loader, empty


class ECPE_Dataset(Dataset):
    def __init__(self, input_file, max_doc_len=75, max_sen_len=60, test=False):
        print('load data_file: {}'.format(input_file))
        self.y_pairs, self.doc_len, self.y_emotion, self.y_cause, self.x_ids, self.x_masks, self.x_types = [], [], [], [], [], [], []
        self.doc_id = []
        self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext',
                                                       cache_dir="model/hfl_chinese-roberta-wwm-ext")
        inputFile = open(input_file, 'r', encoding="utf-8")
        while True:
            line = inputFile.readline()
            if line == '': break
            line = line.strip().split()
            self.doc_id.append(line[0])
            d_len = int(line[1])
            pairs = inputFile.readline().strip().split(", ")
            self.doc_len.append(d_len)
            self.y_pairs.append(pairs)

            y_emotion_tmp, y_cause_tmp, sen_len_tmp, x_id_tmp, x_mask_tmp, x_type_tmp = np.zeros(
                (max_doc_len, 7)), np.zeros(
                (max_doc_len, 7)), np.zeros(
                max_doc_len, dtype=np.int32), np.zeros((max_doc_len, max_sen_len), dtype=np.int32), np.zeros(
                (max_doc_len, max_sen_len), dtype=np.int32), np.zeros((max_doc_len, max_sen_len), dtype=np.int32)
            for i in range(d_len):
                # multiple classe emotion classification
                current_line = inputFile.readline().strip().split(',')
                emotion = current_line[1]

                if emotion == "happiness":
                    y_emotion_tmp[i][0] = 1
                elif emotion == "sadness":
                    y_emotion_tmp[i][1] = 1
                elif emotion == "disgust":
                    y_emotion_tmp[i][2] = 1
                elif emotion == "surprise":
                    y_emotion_tmp[i][3] = 1
                elif emotion == "fear":
                    y_emotion_tmp[i][4] = 1
                elif emotion == "anger":
                    y_emotion_tmp[i][5] = 1
                elif emotion == "null":
                    y_emotion_tmp[i][-1] = 1

                cause = current_line[2]
                if cause == "happiness":
                    y_cause_tmp[i][0] = 1
                elif cause == "sadness":
                    y_cause_tmp[i][1] = 1
                elif cause == "disgust":
                    y_cause_tmp[i][2] = 1
                elif cause == "surprise":
                    y_cause_tmp[i][3] = 1
                elif cause == "fear":
                    y_cause_tmp[i][4] = 1
                elif cause == "anger":
                    y_cause_tmp[i][5] = 1
                elif cause == "null":
                    y_cause_tmp[i][-1] = 1

                sentence = current_line[-1].replace(" ", "")
                inputs = self.tokenizer.encode_plus(
                    sentence,
                    None,
                    add_special_tokens=True,
                    max_length=max_sen_len,
                    padding='max_length',
                    return_token_type_ids=True,
                    truncation=True,
                    return_attention_mask=True
                )

                input_ids = list(inputs['input_ids'])
                input_masks = list(inputs['attention_mask'])
                input_types = list(inputs["token_type_ids"])

                for j in range(len(input_ids)):
                    x_id_tmp[i][j] = input_ids[j]
                    x_mask_tmp[i][j] = input_masks[j]
                    x_type_tmp[i][j] = input_types[j]

            self.y_emotion.append(y_emotion_tmp)
            self.y_cause.append(y_cause_tmp)
            self.x_ids.append(x_id_tmp)
            self.x_masks.append(x_mask_tmp)
            self.x_types.append(x_type_tmp)

        self.y_emotion, self.y_cause, self.x_ids, self.x_masks, self.x_types, self.doc_len = map(
            np.array,
            [self.y_emotion, self.y_cause, self.x_ids, self.x_masks, self.x_types, self.doc_len])
        for var in ['self.y_emotion', 'self.y_cause', 'self.x_ids', 'self.x_masks', 'self.x_types', 'self.doc_len']:
            print('{}.shape {}'.format(var, eval(var).shape))
        print('load data done!\n')

        self.index = [i for i in range(len(self.y_cause))]

    def __getitem__(self, index):
        index = self.index[index]
        feed_list = [self.x_ids[index], self.x_masks[index], self.x_types[index],
                     self.doc_len[index], self.y_emotion[index], self.y_cause[index]]
        return feed_list

    def __len__(self):
        return len(self.x_ids)


class biLSTM(torch.nn.Module):
    def __init__(self, keep_softmax, input_size=opt.n_hidden * 2, hidden_size=opt.n_hidden):
        super(biLSTM, self).__init__()

        self.bert_model = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext",
                                                    cache_dir="model/hfl_chinese-roberta-wwm-ext", return_dict=True)
        a = self.bert_model.config
        self.posbilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.keep_softmax = keep_softmax
        self.nnlayer_pos = nn.Linear(2 * opt.n_hidden, opt.n_class).cuda()
        self.senlayer = nn.Linear(768, 2 * opt.n_hidden, bias=True)

    def forward(self, x_ids, x_masks, x_types):
        x_ids = torch.reshape(x_ids, [-1, opt.max_sen_len])
        x_masks = torch.reshape(x_masks, [-1, opt.max_sen_len])
        x_types = torch.reshape(x_types, [-1, opt.max_sen_len])

        # clause representation shape: (1,75,200)
        s = self.bert_model(
            x_ids,
            attention_mask=x_masks,
            token_type_ids=x_types
        ).pooler_output
        s = torch.reshape(s, [-1, opt.max_doc_len, s.shape[-1]])
        s = self.senlayer(s)
        r_out, (h_n, h_c) = self.posbilstm(s)
        s = r_out
        s1 = torch.reshape(s, [-1, 2 * opt.n_hidden])
        s1 = torch.nn.Dropout(1.0 - self.keep_softmax)(s1)
        pred_emotion = F.softmax(self.nnlayer_pos(s1))
        pred_emotion = torch.reshape(pred_emotion, [-1, opt.max_doc_len, opt.n_class])

        reg = torch.norm(self.nnlayer_pos.weight) + torch.norm(self.nnlayer_pos.bias)
        return pred_emotion, reg


def train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    save_dir = 'pair_data/emotion/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # train
    print_training_info()

    # model
    model = biLSTM(opt.keep_softmax, opt.n_hidden * 2, opt.n_hidden)
    model.to(device)
    model_id = str(uuid4())

    train = "domains/THUCTC_multiple/" + opt.source_domain + ".txt"
    self_train = "domains/THUCTC_multiple/" + opt.target_domain + ".txt"
    test = "domains/THUCTC_multiple/" + opt.target_domain + ".txt"
    edict = {"train": train, "self_train": self_train, "test": test}
    ECPE_data = {x: ECPE_Dataset(edict[x], max_sen_len=opt.max_sen_len, max_doc_len=opt.max_doc_len, test=(x is 'test'))
                 for x in ['train', 'self_train', 'test']}
    train_loader = DataLoader(ECPE_data['train'], batch_size=opt.batch_size, shuffle=True)
    test_loader = DataLoader(ECPE_data['test'], batch_size=opt.batch_size, shuffle=False)

    # train on source domain
    max_f1_emotion = -1.
    best_test_pred_emotion = np.array([])
    for i in range(opt.training_iter):
        start_time, step = time.time(), 1
        for _, data in enumerate(train_loader):
            with torch.autograd.set_detect_anomaly(True):
                x_ids, x_masks, x_types, doc_len, y_emotion, y_cause = data
                x_ids = x_ids.to(device, dtype=torch.long)
                x_masks = x_masks.to(device, dtype=torch.long)
                x_types = x_types.to(device, dtype=torch.long)
                pred_emotion, reg = model(x_ids, x_masks, x_types)
                pred_emotion = pred_emotion.cpu()
                reg = reg.cpu()
                valid_num = torch.sum(doc_len)
                loss_emotion = - torch.sum(y_emotion * torch.log(pred_emotion).double()) / valid_num
                loss_op = loss_emotion * opt.emotion + reg.double() * opt.l2_reg
                optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
                optimizer.zero_grad()
                loss_op = loss_op.cuda()
                loss_op.backward()
                optimizer.step()

                true_y_emo_op = torch.argmax(y_emotion, 2)
                pred_y_emo_op = torch.argmax(pred_emotion, 2)

                true_y_emo_op = true_y_emo_op.cpu()
                pred_y_emo_op = pred_y_emo_op.cpu()

                if step % 50 == 0:
                    print('step {}: train loss {:.4f} '.format(step, loss_op.item()))
                    p, r, f1 = acc_prf(pred_y_emo_op, true_y_emo_op, doc_len)
                    print('emotion predict: train p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1))
                step = step + 1

        test_data_num = 0
        all_test_pred_emotion = torch.tensor([])
        all_test_y_emotion = torch.tensor([])
        all_doc_len = torch.tensor([])

        with torch.no_grad():
            for _, data in enumerate(test_loader):
                x_ids, x_masks, x_types, doc_len, y_emotion, y_cause = data
                x_ids = x_ids.to(device, dtype=torch.long)
                x_masks = x_masks.to(device, dtype=torch.long)
                x_types = x_types.to(device, dtype=torch.long)
                pred_emotion, reg = model(x_ids, x_masks, x_types)
                pred_emotion = pred_emotion.cpu()
                reg = reg.cpu()

                all_test_pred_emotion = torch.cat((all_test_pred_emotion, pred_emotion.float()), 0)
                all_test_y_emotion = torch.cat((all_test_y_emotion, y_emotion.float()), 0)
                all_doc_len = torch.cat((all_doc_len, doc_len.float()), 0)
                valid_num = torch.sum(doc_len).float()
                test_data_num = test_data_num + valid_num

            loss_emotion = -torch.sum(all_test_y_emotion * torch.log(all_test_pred_emotion)) / test_data_num
            loss = loss_emotion * opt.emotion + reg * opt.l2_reg
            print('\nepoch {}: test loss {:.4f} cost time: {:.1f}s\n'.format(i, loss, time.time() - start_time))

            # emotion prediction on target domain
            all_test_y_emotion_op = torch.argmax(all_test_y_emotion, 2)
            all_test_pred_emotion_op = torch.argmax(all_test_pred_emotion, 2)
            all_test_pred_cause_op = torch.full((all_test_pred_emotion_op.shape[0], all_test_pred_emotion_op.shape[1]),
                                                -1)

            p, r, f1 = acc_prf(all_test_pred_emotion_op.numpy(), all_test_y_emotion_op.numpy(),
                               all_doc_len.int().numpy())
            if f1 > max_f1_emotion:
                max_p_emotion, max_r_emotion, max_f1_emotion = p, r, f1
                best_test_pred_emotion = all_test_pred_emotion.numpy()
                save_best_ckp(model.state_dict(), 'ECPE_model/best_emotion_model', model_id)
                print('save the best model......')
            print('emotion predict: test p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1))
            print(
                'max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(max_p_emotion, max_r_emotion, max_f1_emotion))

    # self training on target domain
    self_model = biLSTM(opt.keep_softmax, opt.n_hidden * 2, opt.n_hidden).to(device)
    self_best_test_pred_emotion = best_test_pred_emotion
    max_self_p_emotion, max_self_r_emotion = -1., -1.
    # max_self_f1_emotion = max_f1_emotion
    max_self_f1_emotion = -1.
    empty = False
    num_iter = 0
    while not empty:
        num_iter+=1
        self_best_model = load_ckp(os.path.join('ECPE_model/best_emotion_model', model_id + ".pt"), self_model)
        self_train_loader, empty = generate_self_train_data(ECPE_data['train'], self_best_test_pred_emotion)
        for j in range(5):
            start_time, step = time.time(), 1
            for _, data in enumerate(self_train_loader):
                with torch.autograd.set_detect_anomaly(True):
                    x_ids, x_masks, x_types, doc_len, y_emotion, y_cause = data
                    x_ids = x_ids.to(device, dtype=torch.long)
                    x_masks = x_masks.to(device, dtype=torch.long)
                    x_types = x_types.to(device, dtype=torch.long)
                    pred_emotion, reg = self_best_model(x_ids, x_masks, x_types)
                    pred_emotion = pred_emotion.cpu()
                    reg = reg.cpu()
                    valid_num = torch.sum(doc_len)
                    loss_emotion = - torch.sum(y_emotion * torch.log(pred_emotion).double()) / valid_num
                    loss_op = loss_emotion * opt.emotion + reg.double() * opt.l2_reg
                    optimizer = torch.optim.Adam(self_best_model.parameters(), lr=opt.learning_rate)
                    optimizer.zero_grad()
                    loss_op = loss_op.cuda()
                    loss_op.backward()
                    optimizer.step()

                    true_y_emo_op = torch.argmax(y_emotion, 2)
                    pred_y_emo_op = torch.argmax(pred_emotion, 2)

                    true_y_emo_op = true_y_emo_op.cpu()
                    pred_y_emo_op = pred_y_emo_op.cpu()

                    if step % 50 == 0:
                        print('step {}: self train loss {:.4f} '.format(step, loss_op.item()))
                        p, r, f1 = acc_prf(pred_y_emo_op, true_y_emo_op, doc_len)
                        print(
                            'self emotion predict: train p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1))
                    step = step + 1

            test_data_num = 0
            all_test_pred_emotion = torch.tensor([])
            all_test_y_emotion = torch.tensor([])
            all_doc_len = torch.tensor([])

            with torch.no_grad():
                for _, data in enumerate(test_loader):
                    x_ids, x_masks, x_types, doc_len, y_emotion, y_cause = data
                    x_ids = x_ids.to(device, dtype=torch.long)
                    x_masks = x_masks.to(device, dtype=torch.long)
                    x_types = x_types.to(device, dtype=torch.long)
                    pred_emotion, reg = self_best_model(x_ids, x_masks, x_types)
                    pred_emotion = pred_emotion.cpu()
                    reg = reg.cpu()

                    all_test_pred_emotion = torch.cat((all_test_pred_emotion, pred_emotion.float()), 0)
                    all_test_y_emotion = torch.cat((all_test_y_emotion, y_emotion.float()), 0)
                    all_doc_len = torch.cat((all_doc_len, doc_len.float()), 0)
                    valid_num = torch.sum(doc_len).float()
                    test_data_num = test_data_num + valid_num

                loss_emotion = -torch.sum(all_test_y_emotion * torch.log(all_test_pred_emotion)) / test_data_num
                loss = loss_emotion * opt.emotion + reg * opt.l2_reg
                print('\niteration {} epoch {}: test loss {:.4f} cost time: {:.1f}s\n'.format(num_iter, j, loss,
                                                                                              time.time() - start_time))

                # emotion prediction on target domain
                all_test_y_emotion_op = torch.argmax(all_test_y_emotion, 2)
                all_test_pred_emotion_op = torch.argmax(all_test_pred_emotion, 2)
                all_test_pred_cause_op = torch.full(
                    (all_test_pred_emotion_op.shape[0], all_test_pred_emotion_op.shape[1]),
                    -1)

                p, r, f1 = acc_prf(all_test_pred_emotion_op.numpy(), all_test_y_emotion_op.numpy(),
                                   all_doc_len.int().numpy())
                if f1 > max_self_f1_emotion:
                    max_self_p_emotion, max_self_r_emotion, max_self_f1_emotion = p, r, f1
                    self_best_test_pred_emotion = all_test_pred_emotion.numpy()
                    save_best_ckp(self_best_model.state_dict(), 'ECPE_model/best_emotion_model', model_id)
                    print('save the best self-train model......')
                print('self emotion predict: test p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1))
                print(
                    'self_max_p {:.4f} self_max_r {:.4f} self_max_f1 {:.4f}'.format(max_self_p_emotion,
                                                                                    max_self_r_emotion,
                                                                                    max_self_f1_emotion))
                print(
                    'max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(max_p_emotion, max_r_emotion, max_f1_emotion))

    # pair extraction data
    generate_pair_data(save_dir + opt.target_domain + ".txt", ECPE_data['test'].doc_id, ECPE_data['test'].doc_len,
                       ECPE_data['test'].y_pairs, all_test_pred_cause_op.numpy(),
                       all_test_pred_emotion_op.numpy(), ECPE_data['test'].x_ids)

    print('Optimization Finished!\n')


if __name__ == '__main__':
    train()
