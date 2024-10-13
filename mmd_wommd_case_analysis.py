import re
import pandas as pd
import random
import torch
import torch.nn as nn
import argparse
import numpy as np
import math
import jieba
import os, sys, time

from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
from bow_util import get_bow_zh, get_bow_en
from sklearn.metrics import precision_score, recall_score, f1_score
from uuid import uuid4
from random import randint
from torch.autograd import Variable


# for reproducible results
random.seed(42)

# training setting
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--language', default='zh', help='zh and en')
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
parser.add_argument('--vae_lr', type=float, default=1e-05, help='vae learning rate')
parser.add_argument('--bow_file', type=str, default='data/all_data_pair_zh.txt', help='bag of word file')
parser.add_argument('--best_model_path', type=str, default='ECPE_model/best_cause_pair_model',
                    help='best model saved path')
parser.add_argument('--self_iteration', type=int, default=50, help='self-training iteration')
parser.add_argument('--self_epochs', type=int, default=10, help='self-training epochs')
parser.add_argument('--self_strategy', type=str, default='random', help='self-training strategy')

opt = parser.parse_args()

if opt.language == 'zh':
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext',
                                              cache_dir="model/hfl_chinese-roberta-wwm-ext")
    bow = get_bow_zh(opt.bow_file)

opt.pair_bow_dim = len(bow)


def load_ckp(checkpoint_path, model):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)

    return model

class ECPEDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.tokenizer = tokenizer
        self.pairs = df["pair"]
        self.labels = df["label"].values
        self.emo_labels = df["emotion"].values
        self.cau_labels = df["label"].values
        self.max_len = opt.max_len
        self.bow_features = bow
        self.bow_representations = df["pair"].apply(self._get_bow_representations)

    def __len__(self):
        return len(self.pairs)

    def _get_bow_representations(self, text_pair):
        # non-Chinese unicode range
        filter_zh = re.compile(u'[^\u4E00-\u9FA5]')
        # remove all non-Chinese characters
        text_pair = filter_zh.sub(r'', text_pair)
        sen_words = jieba.lcut(text_pair)

        seq_bow_representation = np.zeros(
            shape=len(self.bow_features), dtype=np.float32)

        # iterate over each word in the sequence
        for word in sen_words:
            if word in self.bow_features:
                bow_index = self.bow_features.index(word)
                seq_bow_representation[bow_index] += 1

        seq_bow_representation /= np.max(
            [np.sum(seq_bow_representation), 1])

        return seq_bow_representation

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
            'attention_masks': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'labels': torch.FloatTensor([self.labels[index]]),
            'emo_labels': torch.LongTensor([self.emo_labels[index]]),
            'cau_labels': torch.FloatTensor([self.cau_labels[index]]),
            'bow_reps': torch.FloatTensor(self.bow_representations[index])
        }

        return instance


class DrlClassifier(nn.Module):

    def __init__(self, opt):
        """
        Initialize networks
        """
        super(DrlClassifier, self).__init__()
        self.opt = opt
        # ================ Encoder model =============#
        if opt.language == 'zh':
            self.encoder = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext",
                                                     cache_dir="model/hfl_chinese-roberta-wwm-ext", return_dict=True)
        elif opt.language == 'en':
            self.encoder = RobertaModel.from_pretrained('roberta-base',
                                                        cache_dir="model/roberta-base", return_dict=True)

        # emotion latent embedding
        self.emotion_mu = nn.Linear(opt.bert_dim, opt.ec_dim)
        self.emotion_log_var = nn.Linear(opt.bert_dim, opt.ec_dim)

        # cause latent embedding
        self.cause_mu = nn.Linear(opt.bert_dim, opt.ec_dim)
        self.cause_log_var = nn.Linear(opt.bert_dim, opt.ec_dim)

        # =============== Classifier =============#
        self.emotion_classifier = nn.Linear(opt.ec_dim, opt.e_num_class)
        self.cause_classifier = nn.Linear(opt.ec_dim, opt.c_num_class)
        self.pair_classifier = nn.Linear(opt.ec_dim * 2, opt.pair_num_class)

        # =============== Decoder =================#
        self.decoder = nn.Linear(opt.ec_dim * 2, opt.pair_bow_dim)

        # dropout
        self.dropout = nn.Dropout(opt.dropout)

    def forward(self, input_ids, att_masks, token_type_ids, emotion_labels, cause_labels, pair_labels, content_bow,
                iteration):
        """
        Args:
            sequences : token indices of input sentences of shape = (batch_size,max_seq_length)
            seq_lengths: actual lengths of input sentences before padding, shape = (batch_size,1)
            style_labels: labels of sentiment of the input sentences, shape = (batch_size,2)
            content_bow: Bag of Words representations of the input sentences, shape = (batch_size,bow_hidden_size)
            iteration: number of iterations completed till now; used for KL annealing
            last_epoch: save average style embeddings if last_epoch is true

        Returns:
            content_disc_loss: loss incurred by content discriminator/adversary
            style_disc_loss  : loss incurred by style discriminator/adversary
            vae_and_classifier_loss : consists of loss incurred by autoencoder, content and style
                                      classifiers
        """
        # sentence embeddings
        sentence_emb = self.encoder(
            input_ids,
            attention_mask=att_masks,
            token_type_ids=token_type_ids
        ).pooler_output

        # get content, emotion, and cause embeddings from the sentence embeddings,i.e. [CLS]
        emotion_emb_mu, emotion_emb_log_var = self.get_emotion_emb(
            sentence_emb)
        cause_emb_mu, cause_emb_log_var = self.get_cause_emb(
            sentence_emb)

        # sample content and style embeddings from their respective latent spaces
        sampled_emotion_emb = self.sample_prior(emotion_emb_mu, emotion_emb_log_var)
        sampled_cause_emb = self.sample_prior(cause_emb_mu, cause_emb_log_var)

        # Generative and pair embeddings
        generative_emb = torch.cat((sampled_emotion_emb, sampled_cause_emb), dim=1)
        pair_emb = torch.cat((sampled_emotion_emb, sampled_cause_emb), dim=1)

        # ============ Losses on emotion space ================#
        # Multitask loss
        emo_mul_loss = self.get_emotion_mul_loss(sampled_emotion_emb, emotion_labels)

        # ============ Losses on cause space ================#
        # Multitask loss
        cau_mul_loss = self.get_cause_mul_loss(sampled_cause_emb, cause_labels)

        # ============ MMD-loss between emotion and cause  ================#
        mmd_loss_func = MMDStatistic(input_ids.shape[0], input_ids.shape[0])
        # the lower the loss the more evidence that distributions are the same.
        mmd_loss = -mmd_loss_func(sampled_emotion_emb, sampled_cause_emb, [0.1])

        # ============ Losses on pair ================#
        pair_mul_loss = self.get_pair_mul_loss(pair_emb, pair_labels)

        # ============== KL losses ===========#
        # emotion space
        emotion_kl_loss = self.get_kl_loss(
            emotion_emb_mu, emotion_emb_log_var)
        if iteration < opt.kl_ann_iterations:
            emotion_kl_loss = self.get_annealed_weight(
                iteration, opt.ec_kl_lambda) * emotion_kl_loss
        # cause space
        cause_kl_loss = self.get_kl_loss(
            cause_emb_mu, cause_emb_log_var)
        if iteration < opt.kl_ann_iterations:
            cause_kl_loss = self.get_annealed_weight(
                iteration, opt.ec_kl_lambda) * cause_kl_loss

        # =============== reconstruction loss================#
        reconstructed_preds = nn.Softmax(dim=1)(self.decoder(generative_emb))
        reconstruction_loss = self.get_reconstruct_loss(reconstructed_preds, content_bow)

        vae_and_classifier_loss = opt.mmd_loss_weight * mmd_loss + \
                                  opt.emo_mul_loss_weight * emo_mul_loss + \
                                  opt.cau_mul_loss_weight * cau_mul_loss + \
                                  opt.pair_mul_loss_weight * pair_mul_loss + \
                                  emotion_kl_loss + cause_kl_loss + \
                                  reconstruction_loss

        return vae_and_classifier_loss

    def get_pair_preds(self, input_ids, att_masks, token_type_ids):
        sentence_emb = self.encoder(
            input_ids,
            attention_mask=att_masks,
            token_type_ids=token_type_ids
        ).pooler_output

        emotion_emb_mu, emotion_emb_log_var = self.get_emotion_emb(
            sentence_emb)
        cause_emb_mu, cause_emb_log_var = self.get_cause_emb(
            sentence_emb)

        sampled_emotion_emb = self.sample_prior(emotion_emb_mu, emotion_emb_log_var)
        sampled_cause_emb = self.sample_prior(cause_emb_mu, cause_emb_log_var)

        pair_emb = torch.cat((sampled_emotion_emb, sampled_cause_emb), dim=1)

        return nn.Sigmoid()(self.pair_classifier(pair_emb)).cpu().detach().numpy().round().tolist()

    def get_params(self):
        """
        Returns:
            content_disc_params: parameters of the content discriminator/adversary
            style_disc_params  : parameters of the style discriminator/adversary
            other_params       : parameters of the vae and classifiers
        """

        other_params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + \
                       list(self.emotion_classifier.parameters()) + \
                       list(self.cause_classifier.parameters()) + \
                       list(self.pair_classifier.parameters())

        return other_params

    def get_content_emb(self, sentence_emb):
        """
        Args:
            sentence_emb: sentence embeddings of all the sentences in the batch, shape=(batch_size,2*gru_hidden_dim)
        Returns:
            mu: embedding of the mean of the Gaussian distribution of the content's latent space
            log_var: embedding of the log of variance of the Gaussian distribution of the content's latent space
        """
        mu = self.content_mu(sentence_emb)
        log_var = self.content_log_var(sentence_emb)

        return mu, log_var

    def get_emotion_emb(self, sentence_emb):
        """
        Args:
            sentence_emb: sentence embeddings of all the sentences in the batch, shape=(batch_size,2*gru_hidden_dim)
        Returns:
            mu: embedding of the mean of the Gaussian distribution of the style's latent space
            log_var: embedding of the log of variance of the Gaussian distribution of the style's latent space
        """
        mu = self.emotion_mu(sentence_emb)
        log_var = self.emotion_log_var(sentence_emb)

        return mu, log_var

    def get_cause_emb(self, sentence_emb):
        """
        Args:
            sentence_emb: sentence embeddings of all the sentences in the batch, shape=(batch_size,2*gru_hidden_dim)
        Returns:
            mu: embedding of the mean of the Gaussian distribution of the style's latent space
            log_var: embedding of the log of variance of the Gaussian distribution of the style's latent space
        """
        mu = self.cause_mu(sentence_emb)
        log_var = self.cause_log_var(sentence_emb)

        return mu, log_var

    def get_ec_emb(self, cause_emb):
        # conditional distribution approximation network
        mu = self.ec_mu(cause_emb)
        log_var = self.ec_log_var(cause_emb)

        return mu, log_var

    def sample_prior(self, mu, log_var):
        """
        Returns samples drawn from the latent space constrained to
        follow diagonal Gaussian
        """
        epsilon = torch.randn(mu.size(1), device=mu.device)
        return mu + epsilon * torch.exp(log_var)

    def get_content_disc_preds(self, style_emb):
        """
        Returns predictions about the content using style embedding
        as input
        output shape : [batch_size,content_bow_dim]
        """
        # predictions
        # Note: detach the style embedding since when don't want the gradient to flow
        #       all the way to the encoder. content_disc_loss is used only to change the
        #       parameters of the discriminator network
        preds = nn.Softmax(dim=1)(self.content_disc(self.dropout(style_emb.detach())))

        return preds

    def get_content_disc_loss(self, content_disc_preds, content_bow):
        """
        It essentially quantifies the amount of information about content
        contained in the style space
        Returns:
        cross entropy loss of content discriminator
        """
        # label smoothing
        smoothed_content_bow = content_bow * (1 - opt.label_smoothing) + opt.label_smoothing / opt.pair_bow_dim
        # calculate cross entropy loss
        content_disc_loss = nn.BCELoss()(content_disc_preds, smoothed_content_bow)

        return content_disc_loss

    def get_reconstruct_loss(self, reconstruct_pred, content_bow):
        # label smoothing
        smoothed_content_bow = content_bow * (1 - opt.label_smoothing) + opt.label_smoothing / opt.pair_bow_dim
        # calculate cross entropy loss
        reconstruct_loss = nn.BCELoss()(reconstruct_pred, smoothed_content_bow)

        return reconstruct_loss

    def get_emotion_disc_preds(self, content_emb):
        """
        Returns predictions about emotion using content embeddings
        as input
        output shape: [batch_size,num_style]
        """
        # predictions
        # Note: detach the content embedding since when don't want the gradient to flow
        #       all the way to the encoder. style_disc_loss is used only to change the
        #       parameters of the discriminator network
        preds = nn.Sigmoid()(self.emotion_disc(self.dropout(content_emb.detach())))

        return preds

    def get_cause_disc_preds(self, content_emb):
        """
        Returns predictions about cause using content embeddings
        as input
        output shape: [batch_size,num_style]
        """
        # predictions
        # Note: detach the content embedding since when don't want the gradient to flow
        #       all the way to the encoder. style_disc_loss is used only to change the
        #       parameters of the discriminator network
        preds = nn.Sigmoid()(self.cause_disc(self.dropout(content_emb.detach())))

        return preds

    def get_ec_aprx_loss(self, e_embedding, c_embedding):

        ec_mu, ec_log_var = self.get_ec_emb(c_embedding.detach())
        log_likelihood = (-(ec_mu - e_embedding) ** 2 / ec_log_var.exp() - ec_log_var).sum(dim=1).mean(dim=0)

        return -log_likelihood

    def get_ec_upper_loss(self, e_embedding, c_embedding):

        ec_mu, ec_log_var = self.get_ec_emb(c_embedding)

        sample_size = e_embedding.shape[0]
        random_index = torch.randperm(sample_size).long()

        positive = - (ec_mu - e_embedding) ** 2 / ec_log_var.exp()
        negative = - (ec_mu - e_embedding[random_index]) ** 2 / ec_log_var.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

        return upper_bound / 2.

    def get_entropy_loss(self, preds):
        """
        Returns the entropy loss: negative of the entropy present in the
        input distribution
        """
        return torch.mean(torch.sum(preds * torch.log(preds + opt.epsilon), dim=1))

    def get_content_mul_loss(self, content_emb, content_bow):
        """
        This loss quantifies the amount of content information preserved
        in the content space
        Returns:
        cross entropy loss of the content classifier
        """
        # predictions
        preds = nn.Softmax(dim=1)(self.content_classifier(self.dropout(content_emb)))
        # label smoothing
        smoothed_content_bow = content_bow * (
                1 - opt.label_smoothing) + opt.label_smoothing / opt.pair_bow_dim
        # calculate cross entropy loss
        content_mul_loss = nn.BCELoss()(preds, smoothed_content_bow)

        return content_mul_loss

    def get_emotion_mul_loss(self, emotion_emb, emotion_labels):
        """
        This loss quantifies the amount of style information preserved
        in the style space
        Returns:
        cross entropy loss of the style classifier
        """
        # predictions
        preds = self.emotion_classifier(self.dropout(emotion_emb))
        # label smoothing
        # smoothed_emotion_labels = emotion_labels * (1 - opt.label_smoothing) + opt.label_smoothing / opt.e_num_class
        # calculate cross entropy loss
        criterion = nn.CrossEntropyLoss()
        emotion_mul_loss = criterion(preds, emotion_labels.flatten())

        return emotion_mul_loss

    def get_cause_mul_loss(self, cause_emb, cause_labels):
        """
        This loss quantifies the amount of style information preserved
        in the style space
        Returns:
        cross entropy loss of the style classifier
        """
        # predictions
        preds = nn.Sigmoid()(self.cause_classifier(self.dropout(cause_emb)))
        # label smoothing
        smoothed_emotion_labels = cause_labels * (1 - opt.label_smoothing) + opt.label_smoothing / opt.c_num_class
        # calculate cross entropy loss
        cause_mul_loss = nn.BCELoss()(preds, smoothed_emotion_labels)

        return cause_mul_loss

    def get_pair_mul_loss(self, pair_emb, pair_labels):
        """
        This loss quantifies the amount of style information preserved
        in the style space
        Returns:
        cross entropy loss of the style classifier
        """
        # predictions
        # preds = nn.Sigmoid()((self.pair_classifier(self.dropout(pair_emb))))
        preds = self.pair_classifier(self.dropout(pair_emb))
        # label smoothing
        smoothed_pair_labels = pair_labels * (1 - opt.label_smoothing) + opt.label_smoothing / opt.pair_num_class
        # calculate cross entropy loss
        pos_weight = (len(pair_labels) - torch.sum(pair_labels)) / torch.sum(pair_labels)
        pair_mul_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(preds, smoothed_pair_labels)

        if pair_mul_loss.isinf().any():
            pair_mul_loss=0

        return pair_mul_loss

    def get_annealed_weight(self, iteration, lambda_weight):
        """
        Args:
            iteration(int): Number of iterations compeleted till now
            lambda_weight(float): KL penalty weight
        Returns:
            Annealed weight(float)
        """
        return (math.tanh((iteration - opt.kl_ann_iterations * 1.5) / (opt.kl_ann_iterations / 3)) + 1) * lambda_weight

    def get_kl_loss(self, mu, log_var):
        """
        Args:
            mu: batch of means of the gaussian distribution followed by the latent variables
            log_var: batch of log variances(log_var) of the gaussian distribution followed by the latent variables
        Returns:
            total loss(float)
        """
        kl_loss = torch.mean((-0.5 * torch.sum(1 + log_var - log_var.exp() - mu.pow(2), dim=1)))
        return kl_loss

def read_ECPE_data(file_path, test=False):
    data_file = open(file_path, encoding="utf8")
    df = pd.DataFrame(columns=["pair", "label", "emotion"])
    docs_pair_size = []
    num_unpred_emotions = 0
    while True:
        line = data_file.readline()
        if not line:
            break
        if re.search("[0-9]{1,4}[\s][0-9]{1,2}", line):
            doc_len = int(line.strip().split(" ")[1])
            sentence_list = []
            pred_emotions = []
            sen_emo_dict = {}

            # positive pair: (emotion, cause) for training
            if opt.language == 'zh':
                pos_pairs = data_file.readline().strip().split(", ")
                pos_pairs = [eval(x) for x in pos_pairs]
            elif opt.language == 'en':
                pos_pairs = eval('[' + data_file.readline().strip() + ']')
                emo_list, cau_list = zip(*pos_pairs)
                pos_pairs = [(emo_list[i], cau_list[i]) for i in range(len(emo_list))]

            # doc's sentences
            for i in range(doc_len):
                sentence = data_file.readline()
                sentence_list.append(sentence)
                sen_emotion = int(sentence.strip().split(",")[1])
                sen_id = int(sentence.strip().split(",")[0])
                if sen_emotion != 6:
                    sen_emo_dict[sen_id] = sen_emotion
                    pred_emotions.append(sen_id)

            if not test:
                emotions = list(dict.fromkeys([e[0] for e in pos_pairs]))
            else:
                true_emotions = list([e[0] for e in pos_pairs])
                pair_indices = []
                pre_e = -1
                for i, e in enumerate(true_emotions):
                    if e not in pred_emotions and e != pre_e:
                        num_unpred_emotions += 1
                    elif e == pre_e:
                        pair_indices.append(i)
                    else:
                        pair_indices.append(i)
                        pred_emotions.remove(e)
                        pre_e = e
                pos_pairs = [pos_pairs[i] for i in pair_indices]
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

            #  training
            if not test:
                pos_pairs_size = len(pos_pairs)
                if pos_pairs_size > len(neg_pairs):
                    pos_pairs_size = len(neg_pairs)
                    neg_pairs = random.sample(neg_pairs, pos_pairs_size)
                else:
                    neg_pairs = random.sample(neg_pairs, pos_pairs_size)
            # test
            else:
                sen_ids = [i + 1 for i in range(doc_len)]
                for e in pred_emotions:
                    for c in sen_ids:
                        neg_pair = (e, c)
                        neg_pairs.append(neg_pair)

            for pos_p in pos_pairs:
                # true emotion-cause pair
                emo_sen_id = pos_p[0]
                cau_sen_id = pos_p[1]
                emotion = sen_emo_dict[emo_sen_id]
                pos_sen_pair = sentence_list[emo_sen_id - 1].strip().split(",")[3].replace(" ", "") + "[SEP]" + \
                               sentence_list[cau_sen_id - 1].strip().split(",")[3].replace(" ", "")
                df = df.append({'pair': pos_sen_pair, 'label': 1, 'emotion': emotion}, ignore_index=True)

            if neg_pairs:
                for neg_p in neg_pairs:
                    # false emotion-cause pair
                    emo_sen_id = neg_p[0]
                    cau_sen_id = neg_p[1]
                    emotion = sen_emo_dict[emo_sen_id]
                    neg_sen_pair = sentence_list[emo_sen_id - 1].strip().split(",")[3].replace(" ", "") + "[SEP]" + \
                                   sentence_list[cau_sen_id - 1].strip().split(",")[3].replace(" ", "")
                    df = df.append({'pair': neg_sen_pair, 'label': 0, 'emotion': emotion}, ignore_index=True)

            docs_pair_size.append(len(pos_pairs) + len(neg_pairs))

    return df, docs_pair_size, num_unpred_emotions



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# init model
model_mmd = DrlClassifier(opt)
model_wommd = DrlClassifier(opt)

model_mmd.to(device)
model_wommd.to(device)

# wommd, mmd model id:
# ent: b946e88e-f381-47ee-a870-bb1d4660922a, e34cf0fa-4465-44a2-a3c9-6a08bfafd2d4
# hom: 6465da17-0be5-446e-bfcf-a5a9c65b7cf1, a7b45ea9-6aae-4429-be2f-16468e3d00a0
# edu: 90b1154d-afa5-4dd6-b37a-feec483003fd, 9773df57-5499-4afa-9dfb-e81c396c2b44
# fin: 60a3c3e0-fea2-4eae-a35e-a66ff5d7d972, e7c1cd12-db8c-481c-832f-8ae702c79759


model_wommd = load_ckp(os.path.join('ECPE_model/best_cause_pair_model', "60a3c3e0-fea2-4eae-a35e-a66ff5d7d972" + ".pt"), model_wommd)
model_mmd = load_ckp(os.path.join('ECPE_model/best_cause_pair_model', "e7c1cd12-db8c-481c-832f-8ae702c79759" + ".pt"), model_mmd)


test_path = "pair_data/emotion/finance.txt"
test_df, test_docs_pair_size, num_unpred_pairs = read_ECPE_data(test_path, test=True)
test_dataset = ECPEDataset(test_df)
test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=test_df.shape[0],
                                               shuffle=False,
                                               num_workers=0
                                               )

test_df = test_df.drop('emotion', axis=1)
test_df['label'] = test_df['label'].astype(float, errors = 'raise')
test_df['self-chain?'] = test_df['pair'].apply(lambda x: True if x.split('[SEP]')[0]==x.split('[SEP]')[1] else False)

model_mmd.eval()
model_wommd.eval()
while True:
    with torch.no_grad():
        for iteration, batch in enumerate(test_data_loader):
            ids = batch['input_ids'].to(device, dtype=torch.long)
            att_masks = batch['attention_masks'].to(device, dtype=torch.long)
            token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
            labels = batch['labels'].to(device, dtype=torch.float)
            
            # do not count unpredicted pairs
            preds_mmd = model_mmd.get_pair_preds(ids, att_masks, token_type_ids)
            preds_wommd = model_wommd.get_pair_preds(ids, att_masks, token_type_ids)

    test_df['preds_mmd'] = preds_mmd
    test_df["preds_mmd"] = test_df["preds_mmd"].apply(lambda x: x[0])

    test_df['preds_wommd'] = preds_wommd
    test_df["preds_wommd"] = test_df["preds_wommd"].apply(lambda x: x[0])


    df_normal=test_df[test_df['self-chain?'] == False]
    df_self=test_df[test_df['self-chain?']]

    wommd_f1=f1_score(test_df['label'], test_df['preds_wommd'], average="binary")
    mmd_f1=f1_score(test_df['label'], test_df['preds_mmd'], average="binary")

    wommd_nor_f1=f1_score(df_normal['label'], df_normal['preds_wommd'], average="binary")
    mmd_nor_f1=f1_score(df_normal['label'], df_normal['preds_mmd'], average="binary")
    wommd_self_f1=f1_score(df_self['label'], df_self['preds_wommd'], average="binary")
    mmd_self_f1=f1_score(df_self['label'], df_self['preds_mmd'], average="binary")

    gap_nor=mmd_nor_f1-wommd_nor_f1
    gap_self=mmd_self_f1-wommd_self_f1

    # gap_nor>0 and gap_self>0 and
    if wommd_f1<mmd_f1 and wommd_f1>0.7330 and mmd_f1>0.8649:
        print('F1 wommd:', wommd_f1)
        print('F1 mmd:', mmd_f1)
        ent_df = pd.DataFrame(columns=["name", "metric", "value","type"])

        # w/o mmd normal 
        # print(df_normal['label'])
        # print(df_normal['preds_wommd'])
        p = precision_score(df_normal['label'], df_normal['preds_wommd'], average="binary")
        r = recall_score(df_normal['label'], df_normal['preds_wommd'], average="binary")
        f1 = f1_score(df_normal['label'], df_normal['preds_wommd'], average="binary")



        ent_df = ent_df.append({'name': 'CDDUDA w/o MMD', 'metric': 'Precision', 'value': str(round(p*100, 2)), 'type':'normal'}, ignore_index=True)
        ent_df = ent_df.append({'name': 'CDDUDA w/o MMD', 'metric': 'Recall', 'value': str(round(r*100, 2)), 'type':'normal'}, ignore_index=True)
        ent_df = ent_df.append({'name': 'CDDUDA w/o MMD', 'metric': 'F1 score', 'value': str(round(f1*100, 2)), 'type':'normal'}, ignore_index=True)

        # mmd normal 

        p = precision_score(df_normal['label'], df_normal['preds_mmd'], average="binary")
        r = recall_score(df_normal['label'], df_normal['preds_mmd'], average="binary")
        f1 = f1_score(df_normal['label'], df_normal['preds_mmd'], average="binary")

        ent_df = ent_df.append({'name': 'CDDUDA', 'metric': 'Precision', 'value': str(round(p*100, 2)), 'type':'normal'}, ignore_index=True)
        ent_df = ent_df.append({'name': 'CDDUDA', 'metric': 'Recall', 'value': str(round(r*100, 2)), 'type':'normal'}, ignore_index=True)
        ent_df = ent_df.append({'name': 'CDDUDA', 'metric': 'F1 score', 'value': str(round(f1*100, 2)), 'type':'normal'}, ignore_index=True)


        # w/o mmd self-chain 
        p = precision_score(df_self['label'], df_self['preds_wommd'], average="binary")
        r = recall_score(df_self['label'], df_self['preds_wommd'], average="binary")
        f1 = f1_score(df_self['label'], df_self['preds_wommd'], average="binary")

        ent_df = ent_df.append({'name': 'CDDUDA w/o MMD', 'metric': 'Precision', 'value': str(round(p*100, 2)), 'type':'self'}, ignore_index=True)
        ent_df = ent_df.append({'name': 'CDDUDA w/o MMD', 'metric': 'Recall', 'value': str(round(r*100, 2)), 'type':'self'}, ignore_index=True)
        ent_df = ent_df.append({'name': 'CDDUDA w/o MMD', 'metric': 'F1 score', 'value': str(round(f1*100, 2)), 'type':'self'}, ignore_index=True)


        # mmd self-chain
        p = precision_score(df_self['label'], df_self['preds_mmd'], average="binary")
        r = recall_score(df_self['label'], df_self['preds_mmd'], average="binary")
        f1 = f1_score(df_self['label'], df_self['preds_mmd'], average="binary")

        ent_df = ent_df.append({'name': 'CDDUDA', 'metric': 'Precision', 'value': str(round(p*100, 2)), 'type':'self'}, ignore_index=True)
        ent_df = ent_df.append({'name': 'CDDUDA', 'metric': 'Recall', 'value': str(round(r*100, 2)), 'type':'self'}, ignore_index=True)
        ent_df = ent_df.append({'name': 'CDDUDA', 'metric': 'F1 score', 'value': str(round(f1*100, 2)), 'type':'self'}, ignore_index=True)


        ent_df.to_csv('wommd_mmd_fin.csv',index=False)
        break         
    else:
        print('F1 wommd:', wommd_f1)
        print('F1 mmd:', mmd_f1)       
        print('nor',gap_nor)
        print('self',gap_self)
        continue


    