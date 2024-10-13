"""
Written by Yujin Huang(Jinx)
Started 2/11/2021 9:34 pm
Last Editted 

Description of the purpose of the code
"""
import re
import pandas as pd
import torch
import shutil
import random
import numpy as np
import os
import time
import sys

from itertools import combinations
from random import randint
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from uuid import uuid4

pd.set_option("display.max_rows", None, "display.max_columns", None)
timestr = time.strftime("%Y%m%d-%H%M%S")
log = open('pair_log' + timestr + '.txt', 'w', buffering=1)
sys.stdout = log

# for reproducible results
random.seed(42)


class ECPEDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.pairs = df["pair"]
        self.labels = df["label"].values
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = str(self.pairs[index])
        inputs = self.tokenizer.encode_plus(
            pair,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        instance = {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'labels': torch.FloatTensor([self.labels[index]])
        }

        return instance


class PairClassifier(torch.nn.Module):
    def __init__(self, dropout):
        super(PairClassifier, self).__init__()
        self.bert_model = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext",
                                                    cache_dir="model/hfl_chinese-roberta-wwm-ext", return_dict=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(768, 1)

    def forward(self, input_ids, att_mask, token_type_ids):
        output = self.bert_model(
            input_ids,
            attention_mask=att_mask,
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output


def read_ECPE_data(file_path, test=False):
    data_file = open(file_path, encoding="utf8")
    df = pd.DataFrame(columns=["pair", "label"])
    docs_pair_size = []

    while True:
        line = data_file.readline()
        if not line:
            break
        if re.search("[0-9]{1,4}[\s][0-9]{1,2}", line):
            doc_len = int(line.strip().split(" ")[1])
            sentence_list = []
            # positive pair: (emotion, cause)
            pos_pairs = data_file.readline().strip().split(", ")
            pos_pairs = [eval(x) for x in pos_pairs]
            emotions = list(dict.fromkeys([e[0] for e in pos_pairs]))
            causes = [c[1] for c in pos_pairs]

            # negative pair: (emotion, non-cause)
            sen_ids = [i + 1 for i in range(doc_len)]
            sen_ids = [i for i in sen_ids if i not in causes]
            neg_pairs = []
            for e in emotions:
                for non_c in sen_ids:
                    neg_pair = (e, non_c)
                    neg_pairs.append(neg_pair)
            #  training proces
            if not test:
                pos_pairs_size = len(pos_pairs)
                if pos_pairs_size > len(neg_pairs):
                    pos_pairs_size = len(neg_pairs)
                    neg_pairs = random.sample(neg_pairs, pos_pairs_size)
                else:
                    neg_pairs = random.sample(neg_pairs, pos_pairs_size)
            # all_poss_pairs = list(combinations(range(0, doc_len), 2))
            # neg_pairs = random.sample([x for x in all_poss_pairs if x not in list(tuple(sorted(t)) for t in pos_pairs)],
            #                           len(pos_pairs))

            for i in range(doc_len):
                sentence = data_file.readline()
                sentence_list.append(sentence)

            for pos_p in pos_pairs:
                # true emotion-cause pair
                emo_sen_id = pos_p[0]
                cau_sen_id = pos_p[1]
                pos_sen_pair = sentence_list[emo_sen_id - 1].strip().split(",")[3].replace(" ", "") + "[SEP]" + \
                               sentence_list[cau_sen_id - 1].strip().split(",")[3].replace(" ", "")
                df = df.append({'pair': pos_sen_pair, 'label': 1}, ignore_index=True)

            if neg_pairs:
                for neg_p in neg_pairs:
                    # false emotion-cause pair
                    emo_sen_id = neg_p[0]
                    cau_sen_id = neg_p[1]
                    neg_sen_pair = sentence_list[emo_sen_id - 1].strip().split(",")[3].replace(" ", "") + "[SEP]" + \
                                   sentence_list[cau_sen_id - 1].strip().split(",")[3].replace(" ", "")
                    df = df.append({'pair': neg_sen_pair, 'label': 0}, ignore_index=True)

            docs_pair_size.append(len(pos_pairs) + len(neg_pairs))

    return df, docs_pair_size


def generate_self_train_data(test_docs_pair_size, test_df, test_loader, model, strategy):
    # obtain prediction
    predicted_df = test_df.copy()
    model.eval()
    with torch.no_grad():
        for batch_index, data in enumerate(test_loader):
            ids = data['input_ids'].to(device, dtype=torch.long)
            att_mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

            outputs = model(ids, att_mask, token_type_ids)
            final_outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
            predicted_df["label"] = final_outputs
            predicted_df["label"] = predicted_df["label"].apply(lambda x: x[0])

    df = pd.DataFrame(columns=["pair", "label"])
    curr_index = 0

    for doc_pair_size in test_docs_pair_size:
        # positive pair: (emotion, cause); negative pair: (emotion, non-cause)
        max_pos_prob = float("-inf")
        max_neg_prob = float("-inf")
        pos_pair = None
        neg_pair = None

        prob_dict = {}

        # pairs in each document
        for i in range(doc_pair_size):
            index = i + curr_index
            pair_label_data = predicted_df.iloc[index]
            prob = pair_label_data["label"]

            # determine whether it is positive or negative
            # strategy 1 threshold: highest score positive > 0.5 and highest score negative < 0.5
            if strategy == "threshold":
                if prob > 0.75 and prob > max_pos_prob:
                    pos_pair = pair_label_data["pair"]
                    max_pos_prob = prob
                elif 0.75 >= prob > max_neg_prob:
                    neg_pair = pair_label_data["pair"]
                    max_neg_prob = prob
            # strategy 2: highest score as positive and random negative
            elif strategy == "random":
                prob_dict[index] = prob
                sort_prob_dict = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
                pos_pair = predicted_df.iloc[sort_prob_dict[0][0]]["pair"]
                if len(sort_prob_dict) == 1: continue
                neg_pair = predicted_df.iloc[sort_prob_dict[randint(1, len(sort_prob_dict) - 1)][0]]["pair"]
            elif strategy == "extreme":
                prob_dict[index] = prob
                sort_prob_dict = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
                pos_pair = predicted_df.iloc[sort_prob_dict[0][0]]["pair"]
                neg_pair = predicted_df.iloc[sort_prob_dict[-1][0]]["pair"]

        curr_index += doc_pair_size

        if pos_pair is not None and neg_pair is not None:
            df = df.append({'pair': pos_pair, 'label': 1}, ignore_index=True)
            df = df.append({'pair': neg_pair, 'label': 0}, ignore_index=True)

    return df


def load_ckp(checkpoint_path, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_path)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch']


def save_ckp(state, is_best, ckpt_path, best_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    # save checkpoint data to the path given, checkpoint_path
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    if not os.path.exists(best_path):
        os.makedirs(best_path)

    ckpt_path = os.path.join(ckpt_path, "cur_model_" + MODEL_ID + ".pt")
    torch.save(state, ckpt_path)
    # if it is a best model, min validation loss
    if is_best:
        # copy that checkpoint file to best path given, best_model_path
        best_path = os.path.join(best_path, "best_model_" + MODEL_ID + ".pt")
        shutil.copyfile(ckpt_path, best_path)


def loss_function(outputs, labels):
    return torch.nn.BCEWithLogitsLoss()(outputs, labels)


def train_model(n_epochs, training_loader, test_loader, model,
                optimizer, checkpoint_path, best_model_path, self_metrics=None, self_train=False):
    # initialize tracker for minimum training loss, precision, recall, and f1 score
    train_loss, max_test_p, max_test_r, max_test_f1 = (0.0 for i in range(4))
    self_max_test_p, self_max_test_r, self_max_test_f1 = (0.0 for i in range(3))

    if self_train:
        self_max_test_p = self_metrics[0]
        self_max_test_r = self_metrics[1]
        self_max_test_f1 = self_metrics[2]

    for epoch in range(1, n_epochs + 1):

        running_loss = 0.0

        # train the model
        model.train()
        print('------------- Epoch {}: Training Start -----------\n'.format(epoch))
        for batch_index, data in enumerate(training_loader):
            ids = data['input_ids'].to(device, dtype=torch.long)
            att_mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.float)

            outputs = model(ids, att_mask, token_type_ids)

            optimizer.zero_grad()
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(data)
            running_loss += loss.item()
            if batch_index % 10 == 9:  # print every 10 mini-batches
                print('[%d, %5d] training loss: %.4f' %
                      (epoch, batch_index + 1, running_loss / 10))
                running_loss = 0.0

        # calculate average losses
        train_loss = train_loss / len(training_loader)
        # valid_loss = valid_loss / len(validation_loader)
        # print("Average Training loss: {}\nAverage Validation loss: {}\n".format(train_loss, valid_loss))
        print("Average Training loss: {}\n".format(train_loss))
        print('------------- Epoch {}: Training End -------------\n'.format(epoch))

        model.eval()
        # running_loss = 0.0

        with torch.no_grad():
            # validate the model
            # print('############# Epoch {}: Validation Start  #############\n'.format(epoch))
            # for batch_idx, data in enumerate(validation_loader):
            #     ids = data['input_ids'].to(device, dtype=torch.long)
            #     att_mask = data['attention_mask'].to(device, dtype=torch.long)
            #     token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            #     labels = data['labels'].to(device, dtype=torch.float)
            #     outputs = model(ids, att_mask, token_type_ids)
            #
            #     loss = loss_function(outputs, labels)
            #     valid_loss += loss.item() * len(data)
            #     running_loss += loss.item()
            #     if batch_index % 10 == 9:  # print every 10 mini-batches
            #         print('[%d, %5d] validation loss: %.4f' %
            #               (epoch + 1, batch_index + 1, running_loss / 10))
            #         running_loss = 0.0
            #
            # print('############# Epoch {}: Validation End    #############\n'.format(epoch))

            # test the model
            print('------------- Test Start -------------------------\n')
            for batch_index, data in enumerate(test_loader):
                ids = data['input_ids'].to(device, dtype=torch.long)
                att_mask = data['attention_mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                labels = data['labels'].to(device, dtype=torch.float)

                outputs = model(ids, att_mask, token_type_ids)
                final_outputs = torch.sigmoid(outputs).cpu().detach().numpy().round().tolist()
                labels = labels.cpu().detach().numpy().tolist()

                # precision, recall, and f1 score
                p = precision_score(labels, final_outputs, average="binary")
                r = recall_score(labels, final_outputs, average="binary")
                f1 = f1_score(labels, final_outputs, average="binary")
                print("current pair precision: {:.4f}, recall: {:.4f}, f1 socre: {:.4f}\n".format(p, r, f1))

                # create checkpoint variable and add important data
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }

                # save checkpoint
                save_ckp(checkpoint, False, checkpoint_path, best_model_path)

                # save the model if test f1 score has increased (source domain)
                if f1 > max_test_f1 and not self_train:
                    print('Test f1 score increased ({:.4f} --> {:.4f}).  Saving model...\n'.format(max_test_f1, f1))
                    # save checkpoint as best model
                    save_ckp(checkpoint, True, checkpoint_path, best_model_path)

                    max_test_p = p
                    max_test_r = r
                    max_test_f1 = f1

                    print(
                        "max pair precision: {:.4f}, recall: {:.4f}, f1 socre: {:.4f}\n".format(max_test_p, max_test_r,
                                                                                                max_test_f1))

                elif f1 > self_max_test_f1 and self_train:
                    print(
                        'Test f1 score increased ({:.4f} --> {:.4f}).  Saving model...\n'.format(self_max_test_f1, f1))

                    save_ckp(checkpoint, True, checkpoint_path, best_model_path)

                    self_max_test_p = p
                    self_max_test_r = r
                    self_max_test_f1 = f1

                    print("max pair precision: {:.4f}, recall: {:.4f}, f1 socre: {:.4f}\n".format(self_max_test_p,
                                                                                                  self_max_test_r,
                                                                                                  self_max_test_f1))

            print('------------- Test end ---------------------------\n')

        print('----------------- Epoch {}  Done ---------------------\n'.format(epoch))
        # obtain the best trained model
    best_model, _, _ = load_ckp(os.path.join(best_model_path, "best_model_" + MODEL_ID + ".pt"), model, optimizer)

    if not self_train:
        return best_model
    else:
        return best_model, self_max_test_p, self_max_test_r, self_max_test_f1


# pre-defined variables
ckpt_path = "ECPE_model/curr_ckpt"
best_model_path = "ECPE_model/best_model"

# set hyperparameters
MAX_LEN = 128
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 64
EPOCHS = 10
SELF_EPOCHS = 10
SELF_ITERATION = 30
LEARNING_RATE = 1e-05
DROPOUT = 0.1
SELF_STRATEGY = "threshold"
MODEL_ID = str(uuid4())

# load train and test data
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext', cache_dir="model/hfl_chinese-roberta-wwm-ext")

train_path = "domains/THUCTC_multiple/society.txt"
test_path = "domains/THUCTC_multiple/entertainment.txt"
train_df, _ = read_ECPE_data(train_path)
test_df, test_docs_pair_size = read_ECPE_data(test_path, test=True)

# train_size = 0.8
# train_df, valid_df = train_test_split(train_df, train_size=train_size, stratify=train_df["label"], shuffle=True,
#                                       random_state=42)

# train_dataset = ECPEDataset(train_df.reset_index(drop=True), tokenizer, MAX_LEN)
# valid_dataset = ECPEDataset(valid_df.reset_index(drop=True), tokenizer, MAX_LEN)
train_dataset = ECPEDataset(train_df, tokenizer, MAX_LEN)
test_dataset = ECPEDataset(test_df, tokenizer, MAX_LEN)

train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=TRAIN_BATCH_SIZE,
                                                shuffle=True,
                                                num_workers=0
                                                )

# val_data_loader = torch.utils.data.DataLoader(valid_dataset,
#                                               batch_size=VALID_BATCH_SIZE,
#                                               shuffle=False,
#                                               num_workers=0
#                                               )

test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=test_df.shape[0],
                                               shuffle=False,
                                               num_workers=0
                                               )

print("------------- Hyperparameter settings --------------")
print("------------- Source domain ------------------------")
print("Epoch: {}\nTraining batch size: {}\nLearning rate: {}\nDropout: {}\nModel id: {}\n".format(EPOCHS,
                                                                                                  TRAIN_BATCH_SIZE,
                                                                                                  LEARNING_RATE,
                                                                                                  DROPOUT, MODEL_ID))
print("------------- Target domain ------------------------")
print(
    "Iteration: {}\nEpoch per iteration: {}\nStrategy: {}\nTraining batch size: {}\nLearning rate: {}\nDropout: {}\nModel id: {}\n".format(
        SELF_ITERATION, SELF_EPOCHS, SELF_STRATEGY, TRAIN_BATCH_SIZE,
        LEARNING_RATE, DROPOUT, MODEL_ID))
print("----------------------------------------------------")
model = PairClassifier(DROPOUT)
model.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
best_trained_model = train_model(EPOCHS, train_data_loader, test_data_loader, model, optimizer, ckpt_path,
                                 best_model_path)

# self-training
print("------------- Self-training start ------------------")
self_best_trained_model = best_trained_model
self_metrics = [0.0, 0.0, 0.0]
for i in range(SELF_ITERATION):
    print("------------- Iteration {} -------------------------".format(i + 1))
    self_train_df = generate_self_train_data(test_docs_pair_size, test_df, test_data_loader, self_best_trained_model,
                                             strategy=SELF_STRATEGY)
    self_train_dataset = ECPEDataset(self_train_df, tokenizer, MAX_LEN)
    self_train_data_loader = torch.utils.data.DataLoader(self_train_dataset,
                                                         batch_size=TRAIN_BATCH_SIZE,
                                                         shuffle=True,
                                                         num_workers=0
                                                         )
    self_best_trained_model, self_p, self_r, self_f1 = train_model(SELF_EPOCHS, self_train_data_loader,
                                                                   test_data_loader,
                                                                   self_best_trained_model,
                                                                   optimizer, ckpt_path,
                                                                   best_model_path, self_metrics=self_metrics,
                                                                   self_train=True)
    self_metrics[0] = self_p
    self_metrics[1] = self_r
    self_metrics[2] = self_f1
print("----------------------------------------------------")