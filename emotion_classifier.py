"""
Written by Yujin Huang(Jinx)
Started 21/11/2021 8:31 pm
Last Editted

Description of the purpose of the code
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import shutil
import os
import re

from torch.autograd import Function
from transformers import BertTokenizer, BertModel
from torchsampler import ImbalancedDatasetSampler
from sklearn.metrics import precision_score, recall_score, f1_score
from uuid import uuid4

# Reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


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


def save_ckp(state, is_best, model_id, ckpt_path, best_path):
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

    ckpt_path = os.path.join(ckpt_path, "cur_emo_classifier_" + model_id + ".pt")
    torch.save(state, ckpt_path)
    # if it is a best model, min validation loss
    if is_best:
        # copy that checkpoint file to best path given, best_model_path
        best_path = os.path.join(best_path, "best_emo_classifier_" + model_id + ".pt")
        shutil.copyfile(ckpt_path, best_path)


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)

    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_

        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads

        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()

        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class EmotionClassifier(nn.Module):
    def __init__(self, opt):
        super(EmotionClassifier, self).__init__()
        # hyperparameters
        self.lexical_feature_dim = opt.lexical_feature_dim  # 768
        self.dropout_rate = opt.dropout_rate  # 0.1
        self.linear_width_l = opt.linear_width_l  # 32
        self.linear_width = opt.linear_width  # 32

        # bert for sentence embeddings
        self.bert_model = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext",
                                                    cache_dir="model/hfl_chinese-roberta-wwm-ext", return_dict=True)
        self.linear_l = nn.Linear(self.lexical_feature_dim, self.linear_width_l)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.batchnorm_l = nn.BatchNorm1d(self.linear_width_l)

        # layers for emotion classifier and domain classifier
        self.linear_1 = nn.Linear(self.linear_width_l, self.linear_width)
        self.linear_2 = nn.Linear(self.linear_width, 7)

    def encoder(self, input_ids, att_mask, token_type_ids):
        x = self.bert_model(
            input_ids,
            attention_mask=att_mask,
            token_type_ids=token_type_ids
        ).pooler_output
        x = self.relu(self.linear_l(x))
        x = self.batchnorm_l(self.dropout(x))
        return x

    def recognizer(self, x):
        x = self.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x

    def forward(self, input_ids, att_mask, token_type_ids):
        x = self.encoder(input_ids, att_mask, token_type_ids)
        x = self.recognizer(x)
        return x


class DomainDiscriminator(nn.Module):
    def __init__(self, opt):
        super(DomainDiscriminator, self).__init__()

        # hyperparameters
        self.bert_model = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext",
                                                    cache_dir="model/hfl_chinese-roberta-wwm-ext", return_dict=True)
        self.linear_width_l = opt.linear_width_l
        self.linear_width = opt.linear_width

        self.grl = GradientReversal(opt.domain_weight)
        self.linear_1 = nn.Linear(self.linear_width_l, self.linear_width)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(self.linear_width, 2)

    def forward(self, x):
        x = self.grl(x)
        x = self.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x


class ECPEDataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext',
                                                       cache_dir="model/hfl_chinese-roberta-wwm-ext")
        self.sentence = df["sentence"]
        self.labels = df["label"].values
        self.max_len = 128

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, index):
        sentence = str(self.sentence[index])
        inputs = self.tokenizer.encode_plus(
            sentence,
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
            'labels': torch.LongTensor([self.labels[index]])
        }

        return instance

    def get_labels(self):
        return self.labels


def read_ECPE_data(file_path):
    data_file = open(file_path, encoding="utf8")
    df = pd.DataFrame(columns=["sentence", "label"])

    while True:
        line = data_file.readline()
        if not line:
            break
        if re.search("[0-9]{1,4}[\s][0-9]{1,2}", line):
            # obtain document length
            doc_len = int(line.strip().split(" ")[1])
            # skip pair: (emotion, cause)
            data_file.readline().strip().split(", ")

            for i in range(doc_len):
                cur_line_label = -1
                cur_line = data_file.readline()
                emotion = cur_line.strip().split(",")[1]
                sentence = cur_line.strip().split(",")[3].replace(" ", "")

                if emotion == "happiness":
                    cur_line_label = 0
                elif emotion == "sadness":
                    cur_line_label = 1
                elif emotion == "disgust":
                    cur_line_label = 2
                elif emotion == "surprise":
                    cur_line_label = 3
                elif emotion == "fear":
                    cur_line_label = 4
                elif emotion == "anger":
                    cur_line_label = 5
                elif emotion == "null":
                    cur_line_label = 6

                df = df.append({'sentence': sentence, 'label': cur_line_label}, ignore_index=True)

    return df


def generate_self_train_data(opt, test_df, test_loader, model):
    # obtain prediction
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    predicted_df = test_df.copy()
    model.eval()
    with torch.no_grad():
        for batch_index, data in enumerate(test_loader):
            ids = data['input_ids'].to(device, dtype=torch.long)
            att_mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

            predictions = model(ids, att_mask, token_type_ids)
            final_outputs = torch.nn.Softmax(dim=1)(predictions).argmax(dim=1).cpu().detach().tolist()
            predicted_df["label"] = final_outputs

    self_dataset = ECPEDataset(predicted_df)
    self_data_loader = torch.utils.data.DataLoader(self_dataset,
                                                   batch_size=opt.batch_size,
                                                   sampler=ImbalancedDatasetSampler(self_dataset),
                                                   num_workers=opt.workers_num
                                                   )
    return self_data_loader


def train_model(opt, training_loader, test_loader, model, model_id, optimizer, self_metrics=None, self_train=False):
    # discriminator = DomainDiscriminator(opt)
    # discriminator.apply(init_weights)
    # for param in discriminator.parameters():
    #     param.requires_grad = True
    # discriminator.to(device)

    # optimizer = torch.optim.Adam(list(emotion_recognizer.parameters())
    #                              + list(discriminator.parameters()),
    #                              lr=opt.learning_rate)

    criterion = torch.nn.CrossEntropyLoss()

    # Train and validate
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    max_test_p, max_test_r, max_test_f1 = (0.0 for i in range(3))

    self_max_test_p, self_max_test_r, self_max_test_f1 = (0.0 for i in range(3))

    if self_train:
        self_max_test_p = self_metrics[0]
        self_max_test_r = self_metrics[1]
        self_max_test_f1 = self_metrics[2]

    for epoch in range(1, opt.epochs_num + 1):

        model.train()
        # discriminator.train()

        tot_emo_loss, tot_dom_loss = (0.0 for i in range(2))
        training_preds = []
        training_labels = []

        for batch_index, data in enumerate(training_loader):
            # training data for emotion classifier
            # source_data = data[0]
            # sou_ids = source_data['input_ids'].to(device, dtype=torch.long)
            # sou_att_mask = source_data['attention_mask'].to(device, dtype=torch.long)
            # sou_token_type_ids = source_data['token_type_ids'].to(device, dtype=torch.long)
            # emo_labels = source_data['labels'].to(device, dtype=torch.long).squeeze()

            sou_ids = data['input_ids'].to(device, dtype=torch.long)
            sou_att_mask = data['attention_mask'].to(device, dtype=torch.long)
            sou_token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            emo_labels = data['labels'].to(device, dtype=torch.long).squeeze()

            # target_data = data[1]
            # tar_ids = target_data['input_ids'].to(device, dtype=torch.long)
            # tar_att_mask = target_data['attention_mask'].to(device, dtype=torch.long)
            # tar_token_type_ids = target_data['token_type_ids'].to(device, dtype=torch.long)

            sou_encoded_x = model.encoder(sou_ids, sou_att_mask, sou_token_type_ids)
            # tar_encoded_x = emotion_recognizer.encoder(tar_ids, tar_att_mask, tar_token_type_ids)

            # training data for domain discriminator
            # dom_encoded_x = torch.cat([sou_encoded_x, tar_encoded_x])
            # dom_labels = torch.cat([torch.ones(sou_encoded_x.shape[0], dtype=torch.int64),
            #                         torch.zeros(tar_encoded_x.shape[0], dtype=torch.int64)]).to(device)

            # two models' predictions
            pred_emo_labels = model.recognizer(sou_encoded_x)
            # pred_dom_labels = discriminator(dom_encoded_x)

            # total loss
            emo_loss = criterion(pred_emo_labels, emo_labels)
            # dom_loss = criterion(pred_dom_labels, dom_labels)
            # loss = emo_loss + dom_loss
            loss = emo_loss

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record loss for analysis
            # tot_emo_loss += emo_loss.item() * len(source_data)
            # tot_dom_loss += dom_loss.item() * dom_encoded_x.shape[0]

            training_labels.extend(emo_labels.cpu().detach().tolist())
            training_preds.extend(torch.nn.Softmax(dim=1)(pred_emo_labels).argmax(dim=1).cpu().detach().tolist())

            # if batch_index + 1 >= num_batches:
            #     break

        # training precision, recall, and f1 score
        training_p = precision_score(training_labels, training_preds, average="micro", labels=[0, 1, 2, 3, 4, 5])
        training_r = recall_score(training_labels, training_preds, average="micro", labels=[0, 1, 2, 3, 4, 5])
        training_f1 = f1_score(training_labels, training_preds, average="micro", labels=[0, 1, 2, 3, 4, 5])

        print('------------- Epoch {}: -----------------------------------------\n'.format(epoch))

        print(
            "current training emotion precision: {:.4f}, recall: {:.4f}, f1 socre: {:.4f}\n".format(training_p,
                                                                                                    training_r,
                                                                                                    training_f1))

        # test precision, recall, and f1 score
        model.eval()
        with torch.no_grad():
            for batch_index, data in enumerate(test_loader):
                ids = data['input_ids'].to(device, dtype=torch.long)
                att_mask = data['attention_mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                labels = data['labels'].to(device, dtype=torch.long)

                predictions = model(ids, att_mask, token_type_ids)
                predictions = torch.nn.Softmax(dim=1)(predictions).argmax(dim=1).cpu().detach().tolist()
                labels = labels.cpu().detach().tolist()

                # precision, recall, and f1 score
                test_p = precision_score(labels, predictions, average="micro", labels=[0, 1, 2, 3, 4, 5])
                test_r = recall_score(labels, predictions, average="micro", labels=[0, 1, 2, 3, 4, 5])
                test_f1 = f1_score(labels, predictions, average="micro", labels=[0, 1, 2, 3, 4, 5])
                print(
                    "current test emotion precision: {:.4f}, recall: {:.4f}, f1 socre: {:.4f}\n".format(test_p, test_r,
                                                                                                        test_f1))

                # create checkpoint variable and add important data
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }

                # save checkpoint
                save_ckp(checkpoint, False, model_id, opt.checkpoint_path, opt.best_model_path)

                # save the model if test f1 score has increased
                if test_f1 > max_test_f1 and not self_train:
                    print(
                        'Test f1 score increased ({:.4f} --> {:.4f}).  Saving model...\n'.format(max_test_f1, test_f1))
                    # save checkpoint as best model
                    save_ckp(checkpoint, True, model_id, opt.checkpoint_path, opt.best_model_path)

                    max_test_p = test_p
                    max_test_r = test_r
                    max_test_f1 = test_f1

                    print(
                        "max test emotion precision: {:.4f}, recall: {:.4f}, f1 socre: {:.4f}\n".format(max_test_p,
                                                                                                        max_test_r,
                                                                                                        max_test_f1))
                elif test_f1 > self_max_test_f1 and self_train:
                    print(
                        'Test f1 score increased ({:.4f} --> {:.4f}).  Saving model...\n'.format(self_max_test_f1,
                                                                                                 test_f1))

                    save_ckp(checkpoint, True, model_id, opt.checkpoint_path, opt.best_model_path)

                    self_max_test_p = test_p
                    self_max_test_r = test_r
                    self_max_test_f1 = test_f1

                    print(
                        "max self emotion precision: {:.4f}, recall: {:.4f}, f1 socre: {:.4f}\n".format(self_max_test_p,
                                                                                                        self_max_test_r,
                                                                                                        self_max_test_f1))

        print('-----------------------------------------------------------------\n')

    best_model, _, _ = load_ckp(os.path.join(opt.best_model_path, "best_emo_classifier_" + model_id + ".pt"),
                                model, optimizer)

    if not self_train:
        return best_model
    else:
        return best_model, self_max_test_p, self_max_test_r, self_max_test_f1


def main():
    parser = argparse.ArgumentParser()

    # Names, paths, logs
    parser.add_argument('--checkpoint_path', default='"ECPE_model/curr_emo_ckpt"',
                        help='relative path to save the current model')
    parser.add_argument('--best_model_path', default='ECPE_model/best_emo_model',
                        help='relative path to save the best model')
    parser.add_argument('--source_domain', default='society', help='society')
    parser.add_argument('--target_domain', default='finance', help='finance or ?')
    parser.add_argument('--verbose', type=bool, default=True, help='True or False')

    # Data parameters
    parser.add_argument('--workers_num', type=int, default=0, help='number of workers for data loading')

    # Training and optimization
    parser.add_argument('--epochs_num', type=int, default=20, help='number of training epochs')
    parser.add_argument('--self_iteration', type=int, default=5, help='15')
    parser.add_argument('--batch_size', type=int, default=32, help='size of a mini-batch')

    parser.add_argument('--learning_rate', type=float, default=1e-05, help='initial learning rate')
    parser.add_argument('--domain_weight', type=float, default=3)

    # Model parameters
    parser.add_argument('--lexical_feature_dim', type=int, default=768)
    parser.add_argument('--linear_width_l', type=int, default=32, help='32')
    parser.add_argument('--linear_width', type=int, default=32, help='32 or 64')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='0.3')

    opt = parser.parse_args()

    if opt.verbose:
        print('------------- Hyperparameter settings ---------------------------\n')
        for arg in vars(opt):
            print(arg + ' = ' + str(getattr(opt, arg)))

    # GPU
    model_id = str(uuid4())
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load train (source + target) and test data (target)
    source_path = "domains/THUCTC_multiple/" + opt.source_domain + ".txt"
    target_path = "domains/THUCTC_multiple/" + opt.target_domain + ".txt"
    source_df = read_ECPE_data(source_path)
    target_df = read_ECPE_data(target_path)

    source_dataset = ECPEDataset(source_df)
    target_dataset = ECPEDataset(target_df)

    source_data_loader = torch.utils.data.DataLoader(source_dataset,
                                                     batch_size=opt.batch_size,
                                                     sampler=ImbalancedDatasetSampler(source_dataset),
                                                     num_workers=opt.workers_num
                                                     )

    # target_data_loader = torch.utils.data.DataLoader(target_dataset,
    #                                                  batch_size=opt.batch_size,
    #                                                  shuffle=True,
    #                                                  num_workers=opt.workers_num
    #                                                  )
    #
    # train_data_loader = list(zip(source_data_loader, target_data_loader))
    # num_batches = min(len(source_data_loader), len(target_data_loader))

    test_df = target_df.copy()
    test_dataset = ECPEDataset(test_df)
    test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=test_df.shape[0],
                                                   shuffle=False,
                                                   num_workers=opt.workers_num
                                                   )

    # model, optimizer and loss function
    emotion_recognizer = EmotionClassifier(opt)
    emotion_recognizer.apply(init_weights)
    for param in emotion_recognizer.parameters():
        param.requires_grad = True
    emotion_recognizer.to(device)

    optimizer = torch.optim.Adam(emotion_recognizer.parameters(), lr=opt.learning_rate)

    best_emo_classifier = train_model(opt, source_data_loader, test_data_loader, emotion_recognizer, model_id,
                                      optimizer,
                                      self_metrics=None, self_train=False)

    print("------------- Self-training start -------------------------------\n")
    self_best_emo_model = best_emo_classifier
    self_metrics = [0.0, 0.0, 0.0]
    opt.epochs_num = 10
    for i in range(opt.self_iteration):
        print("------------- Iteration {} --------------------------------------\n".format(i + 1))
        self_data_loader = generate_self_train_data(opt, test_df, test_data_loader, self_best_emo_model)

        self_best_emo_model, self_p, self_r, self_f1 = train_model(opt, self_data_loader,
                                                                   test_data_loader,
                                                                   self_best_emo_model, model_id,
                                                                   optimizer, self_metrics=self_metrics,
                                                                   self_train=True)
        self_metrics[0] = self_p
        self_metrics[1] = self_r
        self_metrics[2] = self_f1
    print("----------------------------------------------------")


if __name__ == '__main__':
    main()
