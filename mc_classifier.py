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
import faiss

from sentence_transformers import SentenceTransformer
from itertools import combinations
from random import randint
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from uuid import uuid4

pd.set_option("display.max_rows", None, "display.max_columns", None)

# for reproducible results
random.seed(42)


class ECPEDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.triple = df["triple"]
        self.labels = df["label"].values
        self.max_len = max_len

    def __len__(self):
        return len(self.triple)

    def __getitem__(self, index):
        triple = str(self.triple[index])
        inputs = self.tokenizer.encode_plus(
            triple,
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


class CITClassifier(torch.nn.Module):
    def __init__(self, dropout):
        super(CITClassifier, self).__init__()
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
    df = pd.DataFrame(columns=["triple", "label"])
    embedder = SentenceTransformer('cyclone/simcse-chinese-roberta-wwm-ext', cache_folder="model")

    while True:
        line = data_file.readline()
        if not line:
            break
        if re.search("[0-9]{1,4}[\s][0-9]{1,2}", line):
            doc_len = int(line.strip().split(" ")[1])
            sentence_list = []
            # positive triple: (emotion, cause, s)
            pos_pairs = data_file.readline().strip().split(", ")
            pos_pairs = [eval(x) for x in pos_pairs]
            emotions = [e[0] for e in pos_pairs]
            causes = [c[1] for c in pos_pairs]

            # doc's sentences and corresponding ids
            for i in range(doc_len):
                sentence = data_file.readline().strip().split(",")[3].replace(" ", "")
                sentence_list.append(sentence)

            sen_ids = [i + 1 for i in range(doc_len)]
            sen_embeddings = embedder.encode(sentence_list)

            # train data
            if not test:
                # determine whether it is self-chain
                for e, c in zip(emotions, causes):
                    if e == c:
                        # condition on the emotion, self-train emotion sentence is independent of any other sentence except itself
                        pos_sen_triple = sentence_list[e - 1] + "[SEP]" + sentence_list[e - 1] + "[SEP]" + \
                                         sentence_list[e - 1]

                        # knn of e in this doc
                        # neg_sen_id = random.sample([i for i in sen_ids if i != e], 1)[0]
                        index = faiss.IndexFlatL2(sen_embeddings.shape[1])
                        index.add(sen_embeddings)
                        _, neg_index = index.search(np.array([sen_embeddings[e - 1]]), 3)

                        neg_sen_triple = sentence_list[e - 1] + "[SEP]" + \
                                         sentence_list[neg_index[0][2]] + "[SEP]" + sentence_list[e - 1]

                        df = df.append({'triple': pos_sen_triple, 'label': 1}, ignore_index=True)
                        df = df.append({'triple': neg_sen_triple, 'label': 0}, ignore_index=True)

                    else:
                        # chain
                        pos_sen_triple = sentence_list[e - 1] + "[SEP]" + sentence_list[c - 1] + "[SEP]" + \
                                         sentence_list[c - 1]

                        # knn of e in this doc
                        index = faiss.IndexFlatL2(sen_embeddings.shape[1])
                        index.add(sen_embeddings)
                        _, neg_index = index.search(np.array([sen_embeddings[c - 1]]), 3)

                        neg_sen_triple = sentence_list[e - 1] + "[SEP]" + \
                                         sentence_list[neg_index[0][2]] + "[SEP]" + sentence_list[c - 1]

                        df = df.append({'triple': pos_sen_triple, 'label': 1}, ignore_index=True)
                        df = df.append({'triple': neg_sen_triple, 'label': 0}, ignore_index=True)

    return df


def read_pair_data(df_path):
    # predicted pair
    df = pd.read_pickle(df_path)
    pred_pair_indices = df.index[df['label'] == 1].tolist()
    pred_pair = df.loc[df['label'] == 1]

    for i in pred_pair_indices:
        sentences = pred_pair.loc[i, "pair"].split("[SEP]")
        sentences.append(sentences[1])
        triple = "[SEP]".join(sentences)
        pred_pair.loc[i, "pair"] = triple

    test_triple = pred_pair.rename(columns={'pair': 'triple'})

    return test_triple, pred_pair_indices


def generate_self_train_data(best_preds):
    data_file = open(test_path, encoding="utf8")
    pred_df = pd.read_pickle(test_df_path)
    pred_df['label'] = best_preds
    df = pd.DataFrame(columns=["triple", "label"])
    embedder = SentenceTransformer('cyclone/simcse-chinese-roberta-wwm-ext', cache_folder="model")
    doc_index = 0
    curr_index = 0
    pair_size = [26, 16, 12, 16, 19, 9, 9, 11, 6, 17, 13, 15, 16, 21, 22, 10, 15, 13, 18, 18, 13, 16, 12, 13,
                 13, 14, 12, 13, 6, 6, 3, 13, 27, 17, 16, 11, 17, 11, 19, 26, 9, 15, 10, 15, 10, 16, 8, 16, 32,
                 20, 18, 19, 34, 18, 10, 15, 9, 14, 6, 11, 14, 11, 10, 11, 8, 13, 7, 21, 15, 13, 11, 5, 22, 16,
                 9, 15, 11, 20, 19, 12, 69, 20, 11, 11, 17, 16, 20, 17, 17, 18]
    while True:
        line = data_file.readline()
        if not line:
            break
        if re.search("[0-9]{1,4}[\s][0-9]{1,2}", line):
            doc_len = int(line.strip().split(" ")[1])
            sentence_list = []
            # positive triple: (emotion, cause, s)
            pos_pairs = data_file.readline().strip().split(", ")
            # pos_pairs = [eval(x) for x in pos_pairs]
            # emotions = [e[0] for e in pos_pairs]
            # causes = [c[1] for c in pos_pairs]

            # doc's sentences and corresponding ids
            for i in range(doc_len):
                sentence = data_file.readline().strip().split(",")[3].replace(" ", "")
                sentence_list.append(sentence)

            sen_embeddings = embedder.encode(sentence_list)

            index = faiss.IndexFlatL2(sen_embeddings.shape[1])
            index.add(sen_embeddings)

            # predicted ec pair
            for i in range(pair_size[doc_index]):
                i = i + curr_index
                if pred_df.loc[i, "label"] == 1:
                    sentences = pred_df.loc[i, "pair"].split("[SEP]")
                    sentences.append(sentences[1])
                    pos_sen_triple = "[SEP]".join(sentences)

                    # self-chain
                    if sentences[0] == sentences[1]:
                        emo_index = sentence_list.index(sentences[0])

                        _, neg_index = index.search(np.array([sen_embeddings[emo_index]]), 3)

                        neg_sen_triple = sentence_list[emo_index] + "[SEP]" + \
                                         sentence_list[neg_index[0][2]] + "[SEP]" + sentence_list[emo_index]

                        df = df.append({'triple': pos_sen_triple, 'label': 1}, ignore_index=True)
                        df = df.append({'triple': neg_sen_triple, 'label': 0}, ignore_index=True)
                    else:
                        emo_index = sentence_list.index(sentences[0])
                        cau_index = sentence_list.index(sentences[1])

                        _, neg_index = index.search(np.array([sen_embeddings[cau_index]]), 3)

                        neg_sen_triple = sentence_list[emo_index] + "[SEP]" + \
                                         sentence_list[neg_index[0][2]] + "[SEP]" + sentence_list[cau_index]

                        df = df.append({'triple': pos_sen_triple, 'label': 1}, ignore_index=True)
                        df = df.append({'triple': neg_sen_triple, 'label': 0}, ignore_index=True)

            curr_index += pair_size[doc_index]
            doc_index += 1

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

    ckpt_path = os.path.join(ckpt_path, "cur_cit_model_" + MODEL_ID + ".pt")
    torch.save(state, ckpt_path)
    # if it is a best model, min validation loss
    if is_best:
        # copy that checkpoint file to best path given, best_model_path
        best_path = os.path.join(best_path, "best_cit_model_" + MODEL_ID + ".pt")
        shutil.copyfile(ckpt_path, best_path)


def loss_function(outputs, labels):
    return torch.nn.BCEWithLogitsLoss()(outputs, labels)


def train_model(n_epochs, training_loader, test_loader, pred_indices, predictions, true_labels, model,
                optimizer, checkpoint_path, best_model_path, self_metrics=None, self_train=False):
    # initialize tracker for minimum training loss, precision, recall, and f1 score
    max_test_p, max_test_r, max_test_f1 = (0.0 for i in range(3))
    self_max_test_p, self_max_test_r, self_max_test_f1 = (0.0 for i in range(3))
    best_preds = predictions

    if self_train:
        self_max_test_p = self_metrics[0]
        self_max_test_r = self_metrics[1]
        self_max_test_f1 = self_metrics[2]

    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
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
            running_loss += loss.item() * len(data)
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
            # print('------------- Validation Start -------------------------\n')
            # for batch_index, data in enumerate(val_loader):
            #     ids = data['input_ids'].to(device, dtype=torch.long)
            #     att_mask = data['attention_mask'].to(device, dtype=torch.long)
            #     token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            #     labels = data['labels'].to(device, dtype=torch.float)
            #
            #     outputs = model(ids, att_mask, token_type_ids)
            #     final_outputs = torch.sigmoid(outputs).cpu().detach().numpy().round().tolist()
            #     labels = labels.cpu().detach().numpy().tolist()
            #
            #     # precision, recall, and f1 score
            #     p = precision_score(labels, final_outputs, average="binary")
            #     r = recall_score(labels, final_outputs, average="binary")
            #     f1 = f1_score(labels, final_outputs, average="binary")
            #     print("current pair precision: {:.4f}, recall: {:.4f}, f1 socre: {:.4f}\n".format(p, r, f1))

            print('------------- Test Start -------------------------\n')
            for batch_index, data in enumerate(test_loader):
                ids = data['input_ids'].to(device, dtype=torch.long)
                att_mask = data['attention_mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

                outputs = model(ids, att_mask, token_type_ids)
                final_outputs = torch.sigmoid(outputs).cpu().detach().numpy().round().tolist()
                final_outputs = [l[0] for l in final_outputs]

                j = 0
                for i in pred_indices:
                    predictions[i] = final_outputs[j]
                    j += 1

                # precision, recall, and f1 score
                p = precision_score(true_labels, predictions, average="binary")
                r = recall_score(true_labels, predictions, average="binary")
                f1 = f1_score(true_labels, predictions, average="binary")
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
                    best_preds = predictions

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
                    best_preds = predictions

                    print("max pair precision: {:.4f}, recall: {:.4f}, f1 socre: {:.4f}\n".format(self_max_test_p,
                                                                                                  self_max_test_r,
                                                                                                  self_max_test_f1))

        print('----------------- Epoch {}  Done ---------------------\n'.format(epoch))
        # obtain the best trained model
    best_model, _, _ = load_ckp(os.path.join(best_model_path, "best_cit_model_" + MODEL_ID + ".pt"), model, optimizer)

    if not self_train:
        return best_model, best_preds
    else:
        return best_model, self_max_test_p, self_max_test_r, self_max_test_f1, best_preds


# pre-defined variables
ckpt_path = "ECPE_model/curr_cit_ckpt"
best_model_path = "ECPE_model/best_cit_model"

# set hyperparameters
MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
# VALID_BATCH_SIZE = 32
EPOCHS = 1
SELF_EPOCHS = 5
SELF_ITERATION = 10
LEARNING_RATE = 1e-05
DROPOUT = 0.1
# SELF_STRATEGY = "threshold"
MODEL_ID = str(uuid4())

# load train and test data
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext', cache_dir="model/hfl_chinese-roberta-wwm-ext")

train_path = "domains/THUCTC_multiple/society.txt"
test_path = "domains/THUCTC_multiple/finance.txt"
test_df_path = "pair_data/ec_pair/90217c89-92c0-4624-9731-32f3f954f433_pred.pkl"
test_true_path = "pair_data/ec_pair/90217c89-92c0-4624-9731-32f3f954f433_true.pkl"


train_df = read_ECPE_data(train_path)
test_df, test_pred_indices = read_pair_data(test_df_path)

train_size = 0.8
train_df, valid_df = train_test_split(train_df, train_size=train_size, stratify=train_df["label"], shuffle=True,
                                      random_state=42)

train_dataset = ECPEDataset(train_df.reset_index(drop=True), tokenizer, MAX_LEN)
valid_dataset = ECPEDataset(valid_df.reset_index(drop=True), tokenizer, MAX_LEN)
# train_dataset = ECPEDataset(train_df, tokenizer, MAX_LEN)
test_dataset = ECPEDataset(test_df.reset_index(drop=True), tokenizer, MAX_LEN)

train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=TRAIN_BATCH_SIZE,
                                                shuffle=True,
                                                num_workers=0
                                                )

val_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                              batch_size=valid_df.shape[0],
                                              shuffle=False,
                                              num_workers=0
                                              )

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
# print("------------- Target domain ------------------------")
# print(
#     "Iteration: {}\nEpoch per iteration: {}\nStrategy: {}\nTraining batch size: {}\nLearning rate: {}\nDropout: {}\nModel id: {}\n".format(
#         SELF_ITERATION, EPOCHS, SELF_STRATEGY, TRAIN_BATCH_SIZE,
#         LEARNING_RATE, DROPOUT, MODEL_ID))
# print("----------------------------------------------------")
model = CITClassifier(DROPOUT)
model.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
best_trained_model, predictions = train_model(EPOCHS, train_data_loader, test_data_loader,
                                              test_pred_indices,
                                              pd.read_pickle(test_df_path)['label'].tolist(),
                                              pd.read_pickle(test_true_path)['label'].tolist(), model, optimizer,
                                              ckpt_path,
                                              best_model_path)

# self-training
print("------------- Self-training start ------------------")
self_best_trained_model = best_trained_model
best_predictions = predictions
self_metrics = [0.0, 0.0, 0.0]
for i in range(SELF_ITERATION):
    print("------------- Iteration {} -------------------------".format(i + 1))
    self_train_df = generate_self_train_data(best_predictions)
    self_train_dataset = ECPEDataset(self_train_df, tokenizer, MAX_LEN)
    self_train_data_loader = torch.utils.data.DataLoader(self_train_dataset,
                                                         batch_size=TRAIN_BATCH_SIZE,
                                                         shuffle=True,
                                                         num_workers=0
                                                         )
    self_best_trained_model, self_p, self_r, self_f1, best_predictions = train_model(SELF_EPOCHS,
                                                                                     self_train_data_loader,
                                                                                     test_data_loader,
                                                                                     test_pred_indices,
                                                                                     best_predictions,
                                                                                     pd.read_pickle(test_true_path)[
                                                                                         'label'].tolist(),
                                                                                     self_best_trained_model,
                                                                                     optimizer, ckpt_path,
                                                                                     best_model_path,
                                                                                     self_metrics=self_metrics,
                                                                                     self_train=True)
    self_metrics[0] = self_p
    self_metrics[1] = self_r
    self_metrics[2] = self_f1
print("----------------------------------------------------")
