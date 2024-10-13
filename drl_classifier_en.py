"""
Written by Yujin Huang(Jinx)
Started 28/12/2021 10:16 pm
Last Editted 

Description of the purpose of the code
"""
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
from bow_util import get_bow_en
from sklearn.metrics import precision_score, recall_score, f1_score
from uuid import uuid4
from random import randint

# for reproducible results
random.seed(42)

# training setting
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--max_len', type=int, default=128, help='sentence max length')
parser.add_argument('--ec_num_class', type=int, default=1, help='number of emotion or cause class')
parser.add_argument('--pair_num_class', type=int, default=1, help='number of pair class')
parser.add_argument('--ec_dim', type=int, default=24, help='emotion or cause embedding dimension')
parser.add_argument('--con_dim', type=int, default=384, help='content embedding dimension')
parser.add_argument('--pair_bow_dim', type=int, default=23771, help='pair bag of word dimension')
parser.add_argument('--bert_dim', type=int, default=768, help='bert embedding dimension')
parser.add_argument('--kl_ann_iterations', type=int, default=20000, help='kl annealing max iterations')
parser.add_argument('--epochs', type=int, default=10, help='training epochs')
parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
parser.add_argument('--ec_kl_lambda', type=float, default=0.03, help='style kl weight')
parser.add_argument('--con_kl_lambda', type=float, default=0.03, help='content kl weight')
parser.add_argument('--label_smoothing', type=float, default=0.1, help='label smoothing')
parser.add_argument('--con_adv_loss_weight', type=float, default=0.03, help='content adversary loss weight')
parser.add_argument('--ec_adv_loss_weight', type=float, default=1, help='emotion or cause adversary loss weight')
parser.add_argument('--ecce_adv_loss_weight', type=float, default=3, help='emotion and cause adversary loss weight')
parser.add_argument('--con_mul_loss_weight', type=float, default=3, help='content multitask loss weight')
parser.add_argument('--ec_mul_loss_weight', type=float, default=10, help='emotion or cause multitask loss weight')
parser.add_argument('--pair_mul_loss_weight', type=float, default=30, help='pair multitask loss weight')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout probability')
parser.add_argument('--epsilon', type=float, default=1e-8, help='epsilon')
parser.add_argument('--adv_lr', type=float, default=0.001, help='adversary learning rate')
parser.add_argument('--vae_lr', type=float, default=1e-05, help='vae learning rate')
parser.add_argument('--bow_file', type=str, default='data/all_data_pair_en.txt', help='bag of word file')
parser.add_argument('--best_model_path', type=str, default='ECPE_model/best_drl_model', help='best model saved path')
parser.add_argument('--self_iteration', type=int, default=30, help='self-training iteration')
parser.add_argument('--self_epochs', type=int, default=10, help='self-training epochs')
parser.add_argument('--self_strategy', type=str, default='random', help='self-training strategy')

# parser.add_argument('--embedding_dim', type=int, default=200, help='word embedding dimension')
# parser.add_argument('--embedding_file', type=str, default='w2v_200.txt', help='word embedding file')
# parser.add_argument('--doc_keywords', type=str, default='all_data_pair.txt', help='document keywords file')


opt = parser.parse_args()
opt.model_id = str(uuid4())
bow = get_bow_en(opt.bow_file)
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir="model/bert-base-uncased")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base", cache_dir="model/roberta-base")
opt.pair_bow_dim = len(bow)

timestr = time.strftime("%Y%m%d-%H%M%S")
log = open('drl_en_log_' + timestr + '.txt', 'w', buffering=1)
sys.stdout = log
sys.stderr = log


class ECPEDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.tokenizer = tokenizer
        self.pairs = df["pair"]
        self.labels = df["label"].values
        self.emo_labels = np.ones(df["label"].values.size, dtype=np.int)
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
            'emo_labels': torch.FloatTensor([self.emo_labels[index]]),
            'cau_labels': torch.FloatTensor([self.cau_labels[index]]),
            'bow_reps': torch.FloatTensor(self.bow_representations[index])
        }

        return instance


class DrlClassifier(nn.Module):
    """
    Model architecture defined according to the paper
    'Disentangled Representation Learning for Non-Parallel Text Style Transfer'
    https://www.aclweb.org/anthology/P19-1041.pdf

    """

    def __init__(self, opt):
        """
        Initialize networks
        """
        super(DrlClassifier, self).__init__()
        self.opt = opt,
        # ================ Encoder model =============#
        # self.encoder = BertModel.from_pretrained("bert-base-uncased", cache_dir="model/bert-base-uncased",
        #                                          return_dict=True)
        self.encoder = RobertaModel.from_pretrained("roberta-base", cache_dir="model/roberta-base",
                                                 return_dict=True)

        # content latent embedding
        self.content_mu = nn.Linear(opt.bert_dim, opt.con_dim)
        self.content_log_var = nn.Linear(opt.bert_dim, opt.con_dim)

        # emotion latent embedding
        self.emotion_mu = nn.Linear(opt.bert_dim, opt.ec_dim)
        self.emotion_log_var = nn.Linear(opt.bert_dim, opt.ec_dim)

        # cause latent embedding
        self.cause_mu = nn.Linear(opt.bert_dim, opt.ec_dim)
        self.cause_log_var = nn.Linear(opt.bert_dim, opt.ec_dim)

        # =============== Discriminator/adversary============#
        self.emotion_disc = nn.Linear(opt.con_dim, opt.ec_num_class)
        self.content_disc = nn.Linear(opt.ec_dim, opt.pair_bow_dim)
        self.cause_disc = nn.Linear(opt.con_dim, opt.ec_num_class)
        self.ec_disc = nn.Linear(opt.ec_dim, opt.ec_num_class)
        self.ce_disc = nn.Linear(opt.ec_dim, opt.ec_num_class)

        # =============== Classifier =============#
        self.content_classifier = nn.Linear(opt.con_dim, opt.pair_bow_dim)
        self.emotion_classifier = nn.Linear(opt.ec_dim, opt.ec_num_class)
        self.cause_classifier = nn.Linear(opt.ec_dim, opt.ec_num_class)
        self.pair_classifier = nn.Linear(opt.ec_dim * 2, opt.pair_num_class)

        # =============== Decoder =================#
        self.decoder = nn.Linear(opt.ec_dim * 2 + opt.con_dim, opt.pair_bow_dim)

        # ============== Average label embedding ======#
        # Used during inference to transfer style.
        # Each element of the dict consists of average of latent style embeddings
        # of all the sentences of that particular label/style.
        # 0 -> negative, 1 -> positive
        self.avg_style_emb = {
            0: torch.zeros(opt.ec_dim),
            1: torch.zeros(opt.ec_dim)
        }

        # Used to maintain a running average
        self.num_neg_styles = 0
        self.num_pos_styles = 0

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
        content_emb_mu, content_emb_log_var = self.get_content_emb(
            sentence_emb)
        emotion_emb_mu, emotion_emb_log_var = self.get_emotion_emb(
            sentence_emb)
        cause_emb_mu, cause_emb_log_var = self.get_cause_emb(
            sentence_emb)

        # sample content and style embeddings from their respective latent spaces
        sampled_content_emb = self.sample_prior(content_emb_mu, content_emb_log_var)
        sampled_emotion_emb = self.sample_prior(emotion_emb_mu, emotion_emb_log_var)
        sampled_cause_emb = self.sample_prior(cause_emb_mu, cause_emb_log_var)

        # Generative and pair embeddings
        generative_emb = torch.cat((sampled_emotion_emb, sampled_cause_emb, sampled_content_emb), dim=1)
        pair_emb = torch.cat((sampled_emotion_emb, sampled_cause_emb), dim=1)

        # =========== Losses on content space =============#
        # Discriminator Loss
        content_disc_preds_emo = self.get_content_disc_preds(sampled_emotion_emb)
        content_disc_loss_emo = self.get_content_disc_loss(content_disc_preds_emo, content_bow)
        content_disc_preds_cau = self.get_content_disc_preds(sampled_cause_emb)
        content_disc_loss_cau = self.get_content_disc_loss(content_disc_preds_cau, content_bow)
        # adversarial entropy
        content_entropy_loss_emo = self.get_entropy_loss(content_disc_preds_emo)
        content_entropy_loss_cau = self.get_entropy_loss(content_disc_preds_cau)
        # Multitask loss
        content_mul_loss = self.get_content_mul_loss(sampled_content_emb, content_bow)

        # ============ Losses on emotion space ================#
        # Discriminator loss
        emotion_disc_preds = self.get_emotion_disc_preds(sampled_content_emb)
        emotion_disc_loss = self.get_ec_disc_loss(emotion_disc_preds, emotion_labels)
        ec_disc_preds = self.get_ec_disc_preds(sampled_cause_emb)
        ec_disc_loss = self.get_ec_disc_loss(ec_disc_preds, emotion_labels)
        # adversarial entropy
        emotion_entropy_loss = self.get_entropy_loss(emotion_disc_preds)
        ec_entropy_loss = self.get_entropy_loss(ec_disc_preds)
        # Multitask loss
        emo_mul_loss = self.get_emotion_mul_loss(sampled_emotion_emb, emotion_labels)

        # ============ Losses on cause space ================#
        # Discriminator loss
        cause_disc_preds = self.get_cause_disc_preds(sampled_content_emb)
        cause_disc_loss = self.get_ec_disc_loss(cause_disc_preds, cause_labels)
        ce_disc_preds = self.get_ce_disc_preds(sampled_emotion_emb)
        ce_disc_loss = self.get_ec_disc_loss(ce_disc_preds, cause_labels)
        # adversarial entropy
        cause_entropy_loss = self.get_entropy_loss(cause_disc_preds)
        ce_entropy_loss = self.get_entropy_loss(ce_disc_preds)
        # Multitask loss
        cau_mul_loss = self.get_cause_mul_loss(sampled_cause_emb, cause_labels)

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
        # Content space
        content_kl_loss = self.get_kl_loss(
            content_emb_mu, content_emb_log_var)
        if iteration < opt.kl_ann_iterations:
            content_kl_loss = self.get_annealed_weight(
                iteration, opt.con_kl_lambda) * content_kl_loss

        # =============== reconstruction loss================#
        reconstructed_preds = nn.Softmax(dim=1)(self.decoder(generative_emb))
        reconstruction_loss = self.get_reconstruct_loss(reconstructed_preds, content_bow)

        # ================ total weighted loss ==========#
        # print("content_entropy_loss_emo", content_entropy_loss_emo)
        # print("content_entropy_loss_cau", content_entropy_loss_cau)
        # print("emotion_entropy_loss", emotion_entropy_loss)
        # print("cause_entropy_loss", cause_entropy_loss)
        # print("ec_entropy_loss", ec_entropy_loss)
        # print("ce_entropy_loss", ce_entropy_loss)
        # print("emo_mul_loss", emo_mul_loss)
        # print("cau_mul_loss", cau_mul_loss)
        # print("content_mul_loss", content_mul_loss)
        # print("pair_mul_loss", pair_mul_loss)
        # print("emotion_kl_loss", emotion_kl_loss)
        # print("cause_kl_loss", cause_kl_loss)
        # print("content_kl_loss", content_kl_loss)
        # print("reconstruction_loss", reconstruction_loss)

        vae_and_classifier_loss = opt.con_adv_loss_weight * (content_entropy_loss_emo + content_entropy_loss_cau) + \
                                  opt.ec_adv_loss_weight * (emotion_entropy_loss + cause_entropy_loss) + \
                                  opt.ecce_adv_loss_weight * (ec_entropy_loss + ce_entropy_loss) + \
                                  opt.ec_mul_loss_weight * (emo_mul_loss + cau_mul_loss) + \
                                  opt.con_mul_loss_weight * content_mul_loss + \
                                  opt.pair_mul_loss_weight * pair_mul_loss + \
                                  emotion_kl_loss + cause_kl_loss + content_kl_loss + \
                                  reconstruction_loss

        return content_disc_loss_emo, content_disc_loss_cau, emotion_disc_loss, ec_disc_loss, cause_disc_loss, ce_disc_loss, vae_and_classifier_loss

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

        return self.pair_classifier(pair_emb)

        # return nn.Sigmoid()(self.pair_classifier(pair_emb)).cpu().detach().numpy().round().tolist()

    def get_params(self):
        """
        Returns:
            content_disc_params: parameters of the content discriminator/adversary
            style_disc_params  : parameters of the style discriminator/adversary
            other_params       : parameters of the vae and classifiers
        """

        content_disc_params = self.content_disc.parameters()
        emotion_disc_params = self.emotion_disc.parameters()
        cause_disc_params = self.cause_disc.parameters()
        ec_disc_params = self.ec_disc.parameters()
        ce_disc_params = self.ce_disc.parameters()
        other_params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + \
                       list(self.emotion_classifier.parameters()) + \
                       list(self.cause_classifier.parameters()) + \
                       list(self.pair_classifier.parameters()) + \
                       list(self.content_classifier.parameters())

        return content_disc_params, emotion_disc_params, cause_disc_params, ec_disc_params, ce_disc_params, other_params

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

    def get_ec_disc_preds(self, cause_emb):
        """
        Returns predictions about emotion using cause embeddings
        as input
        output shape: [batch_size,num_style]
        """
        # predictions
        # Note: detach the content embedding since when don't want the gradient to flow
        #       all the way to the encoder. style_disc_loss is used only to change the
        #       parameters of the discriminator network
        preds = nn.Sigmoid()(self.ec_disc(self.dropout(cause_emb.detach())))

        return preds

    def get_ce_disc_preds(self, emotion_emb):
        """
        Returns predictions about cause using emotion embeddings
        as input
        output shape: [batch_size,num_style]
        """
        # predictions
        # Note: detach the content embedding since when don't want the gradient to flow
        #       all the way to the encoder. style_disc_loss is used only to change the
        #       parameters of the discriminator network
        preds = nn.Sigmoid()(self.ce_disc(self.dropout(emotion_emb.detach())))

        return preds

    def get_ec_disc_loss(self, ec_disc_preds, ec_labels):
        """
        It essentially quantifies the amount of information about style
        contained in the content space
        Returns:
        cross entropy loss of style discriminator
        """
        # label smoothing
        smoothed_ec_labels = ec_labels * (1 - opt.label_smoothing) + opt.label_smoothing / opt.ec_num_class
        # calculate cross entropy loss
        ec_disc_loss = nn.BCELoss()(ec_disc_preds, smoothed_ec_labels)

        return ec_disc_loss

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
        preds = nn.Sigmoid()(self.emotion_classifier(self.dropout(emotion_emb)))
        # label smoothing
        smoothed_emotion_labels = emotion_labels * (1 - opt.label_smoothing) + opt.label_smoothing / opt.ec_num_class
        # calculate cross entropy loss
        emotion_mul_loss = nn.BCELoss()(preds, smoothed_emotion_labels)

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
        smoothed_emotion_labels = cause_labels * (1 - opt.label_smoothing) + opt.label_smoothing / opt.ec_num_class
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
        smoothed_pair_labels = pair_labels * (1 - opt.label_smoothing) + opt.label_smoothing / opt.ec_num_class
        # calculate cross entropy loss
        pos_weight = (len(pair_labels) - torch.sum(pair_labels)) / torch.sum(pair_labels)
        pair_mul_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(preds, smoothed_pair_labels)

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


def save_ckp(state, ckpt_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    # save checkpoint data to the path given, checkpoint_path
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    ckpt_path = os.path.join(ckpt_path, "best_drl_en_model_" + opt.model_id + ".pt")
    torch.save(state, ckpt_path)


# def read_ECPE_data(file_path, test=False):
#     data_file = open(file_path, encoding="utf8")
#     df = pd.DataFrame(columns=["pair", "label"])
#     docs_pair_size = []
#     num_unpred_emotions = 0
#     while True:
#         line = data_file.readline()
#         if not line:
#             break
#         if re.search("[0-9]{1,4}[\s][0-9]{1,2}", line):
#             doc_len = int(line.strip().split(" ")[1])
#             sentence_list = []
#             pred_emotions = []
#
#             # positive pair: (emotion, cause) for training
#             pos_pairs = data_file.readline().strip().split(", ")
#             pos_pairs = [eval(x) for x in pos_pairs]
#             # doc's sentences
#             for i in range(doc_len):
#                 sentence = data_file.readline()
#                 sentence_list.append(sentence)
#                 if test:
#                     sen_emotion = int(sentence.strip().split(",")[1])
#                     sen_id = int(sentence.strip().split(",")[0])
#                     if sen_emotion != 6:
#                         pred_emotions.append(sen_id)
#
#             if not test:
#                 emotions = list(dict.fromkeys([e[0] for e in pos_pairs]))
#             else:
#                 true_emotions = list([e[0] for e in pos_pairs])
#                 pair_indices = []
#                 pre_e = -1
#                 for i, e in enumerate(true_emotions):
#                     if e not in pred_emotions and e != pre_e:
#                         num_unpred_emotions += 1
#                     elif e == pre_e:
#                         pair_indices.append(i)
#                     else:
#                         pair_indices.append(i)
#                         pred_emotions.remove(e)
#                         pre_e = e
#                 pos_pairs = [pos_pairs[i] for i in pair_indices]
#                 emotions = list(dict.fromkeys([e[0] for e in pos_pairs]))
#
#             causes = [c[1] for c in pos_pairs]
#
#             # negative pair: (emotion, non-cause)
#             sen_ids = [i + 1 for i in range(doc_len)]
#             sen_ids = [i for i in sen_ids if i not in causes]
#             neg_pairs = []
#             for e in emotions:
#                 for non_c in sen_ids:
#                     neg_pair = (e, non_c)
#                     neg_pairs.append(neg_pair)
#
#             #  training
#             if not test:
#                 pos_pairs_size = len(pos_pairs)
#                 if pos_pairs_size > len(neg_pairs):
#                     pos_pairs_size = len(neg_pairs)
#                     neg_pairs = random.sample(neg_pairs, pos_pairs_size)
#                 else:
#                     neg_pairs = random.sample(neg_pairs, pos_pairs_size)
#             # test
#             else:
#                 sen_ids = [i + 1 for i in range(doc_len)]
#                 for e in pred_emotions:
#                     for c in sen_ids:
#                         neg_pair = (e, c)
#                         neg_pairs.append(neg_pair)
#
#             for pos_p in pos_pairs:
#                 # true emotion-cause pair
#                 emo_sen_id = pos_p[0]
#                 cau_sen_id = pos_p[1]
#                 pos_sen_pair = sentence_list[emo_sen_id - 1].strip().split(",")[3].replace(" ", "") + "[SEP]" + \
#                                sentence_list[cau_sen_id - 1].strip().split(",")[3].replace(" ", "")
#                 df = df.append({'pair': pos_sen_pair, 'label': 1}, ignore_index=True)
#
#             if neg_pairs:
#                 for neg_p in neg_pairs:
#                     # false emotion-cause pair
#                     emo_sen_id = neg_p[0]
#                     cau_sen_id = neg_p[1]
#                     neg_sen_pair = sentence_list[emo_sen_id - 1].strip().split(",")[3].replace(" ", "") + "[SEP]" + \
#                                    sentence_list[cau_sen_id - 1].strip().split(",")[3].replace(" ", "")
#                     df = df.append({'pair': neg_sen_pair, 'label': 0}, ignore_index=True)
#
#             docs_pair_size.append(len(pos_pairs) + len(neg_pairs))
#
#     return df, docs_pair_size, num_unpred_emotions

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
            # pos_pairs = data_file.readline().strip().split(", ")
            # pos_pairs = [eval(x) for x in pos_pairs]
            pos_pairs = eval('[' + data_file.readline().strip() + ']')
            emo_list, cau_list = zip(*pos_pairs)
            pos_pairs = [(emo_list[i], cau_list[i]) for i in range(len(emo_list))]
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
            att_masks = data['attention_masks'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

            final_outputs = model.get_pair_preds(ids, att_masks, token_type_ids)
            predicted_df["label"] = torch.sigmoid(final_outputs).cpu().detach().numpy().tolist()
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
                if prob > 0.5 and prob > max_pos_prob:
                    pos_pair = pair_label_data["pair"]
                    max_pos_prob = prob
                    print('positive',prob)
                elif 0.5 >= prob > max_neg_prob:
                    neg_pair = pair_label_data["pair"]
                    max_neg_prob = prob
                    print('negative',prob)
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


def train(train_loader, test_loader, model, optimizers, device, num_unpred_pairs=0, self_metrics=None,
          self_train=False):
    # obtain all optimizers
    content_disc_opt, emotion_disc_opt, cause_disc_opt, ec_disc_opt, ce_disc_opt, vae_and_cls_opt = optimizers

    # initialize tracker for minimum training loss, precision, recall, and f1 score
    max_test_p, max_test_r, max_test_f1 = (0.0 for i in range(3))
    self_max_test_p, self_max_test_r, self_max_test_f1 = (0.0 for i in range(3))

    if self_train:
        self_max_test_p = self_metrics[0]
        self_max_test_r = self_metrics[1]
        self_max_test_f1 = self_metrics[2]
        epochs = opt.self_epochs
    else:
        epochs = opt.epochs

    for epoch in range(1, epochs + 1):
        training_loss = 0
        running_loss = 0

        model.train()
        print('\n############ Epoch {}: Training Start ############\n'.format(epoch))
        for iteration, batch in enumerate(train_loader):
            ids = batch['input_ids'].to(device, dtype=torch.long)
            att_masks = batch['attention_masks'].to(device, dtype=torch.long)
            token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
            labels = batch['labels'].to(device, dtype=torch.float)
            emo_labels = batch['emo_labels'].to(device, dtype=torch.float)
            cau_labels = batch['cau_labels'].to(device, dtype=torch.float)
            bow_reps = batch['bow_reps'].to(device, dtype=torch.float)

            content_disc_loss_emo, content_disc_loss_cau, \
            emotion_disc_loss, ec_disc_loss, \
            cause_disc_loss, ce_disc_loss, \
            vae_and_cls_loss = model(ids, att_masks, token_type_ids, emo_labels, cau_labels, labels, bow_reps,
                                     iteration)

            with torch.autograd.set_detect_anomaly(True):
                # Update Adversary/Discriminator parameters
                content_disc_opt.zero_grad()
                (content_disc_loss_emo + content_disc_loss_cau).backward(retain_graph=True)

                # update emotion discriminator parameters
                emotion_disc_opt.zero_grad()
                emotion_disc_loss.backward(retain_graph=True)

                ec_disc_opt.zero_grad()
                ec_disc_loss.backward(retain_graph=True)

                # update cause discriminator parameters
                cause_disc_opt.zero_grad()
                cause_disc_loss.backward(retain_graph=True)

                ce_disc_opt.zero_grad()
                ce_disc_loss.backward(retain_graph=True)

                # update VAE and classifier parameters
                vae_and_cls_opt.zero_grad()
                vae_and_cls_loss.backward()

                content_disc_opt.step()
                emotion_disc_opt.step()
                ec_disc_opt.step()
                cause_disc_opt.step()
                ce_disc_opt.step()
                vae_and_cls_opt.step()

            # total loss
            training_loss += content_disc_loss_emo.item() + content_disc_loss_cau.item() + \
                             emotion_disc_loss.item() + ec_disc_loss.item() + \
                             cause_disc_loss.item() + ce_disc_loss.item() + \
                             vae_and_cls_loss.item()

            running_loss += content_disc_loss_emo.item() + content_disc_loss_cau.item() + \
                            emotion_disc_loss.item() + ec_disc_loss.item() + \
                            cause_disc_loss.item() + ce_disc_loss.item() + \
                            vae_and_cls_loss.item()

            # print running loss every 10 mini-batches
            if iteration % 10 == 9:
                print('[%d, %5d] training loss: %.4f' %
                      (epoch, iteration + 1, running_loss / 10))
                running_loss = 0.0

        training_loss = running_loss / len(train_loader)
        print('\ntraining loss {}\n'.format(training_loss))

        model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(test_loader):
                ids = batch['input_ids'].to(device, dtype=torch.long)
                att_masks = batch['attention_masks'].to(device, dtype=torch.long)
                token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
                labels = batch['labels'].to(device, dtype=torch.float)

                labels = labels.cpu().detach().numpy().tolist()
                preds = model.get_pair_preds(ids, att_masks, token_type_ids)
                preds = nn.Sigmoid()(preds).cpu().detach().numpy().round().tolist()
                # append the number of unpredicted pairs
                labels += [[1]] * num_unpred_pairs
                preds += [[0]] * num_unpred_pairs

                # precision, recall, and f1 score
                p = precision_score(labels, preds, average="binary")
                r = recall_score(labels, preds, average="binary")
                f1 = f1_score(labels, preds, average="binary")
                print("current test pair precision: {:.4f}, recall: {:.4f}, f1 socre: {:.4f}\n".format(p, r, f1))

                # create checkpoint variable and add important data
                checkpoint = model.state_dict()

                # save the model if test f1 score has increased (source domain)
                if f1 > max_test_f1 and not self_train:
                    print('Test f1 score increased ({:.4f} --> {:.4f}).  Saving model...\n'.format(max_test_f1, f1))
                    # save checkpoint as best model
                    save_ckp(checkpoint, opt.best_model_path)

                    max_test_p = p
                    max_test_r = r
                    max_test_f1 = f1

                    print(
                        "max pair precision: {:.4f}, recall: {:.4f}, f1 socre: {:.4f}\n".format(max_test_p, max_test_r,
                                                                                                max_test_f1))

                elif f1 > self_max_test_f1 and self_train:
                    print(
                        'test f1 score increased ({:.4f} --> {:.4f}).  Saving model...\n'.format(self_max_test_f1, f1))

                    save_ckp(checkpoint, opt.best_model_path)

                    self_max_test_p = p
                    self_max_test_r = r
                    self_max_test_f1 = f1

                if not self_train:
                    print(
                        "max test pair precision: {:.4f}, recall: {:.4f}, f1 socre: {:.4f}\n".format(max_test_p,
                                                                                                     max_test_r,
                                                                                                     max_test_f1))
                else:
                    print("self max test pair precision: {:.4f}, recall: {:.4f}, f1 socre: {:.4f}\n".format(
                        self_max_test_p,
                        self_max_test_r,
                        self_max_test_f1))

    # obtain the best trained model
    best_model = load_ckp(os.path.join(opt.best_model_path, "best_drl_en_model_" + opt.model_id + ".pt"), model)

    if not self_train:
        return best_model
    else:
        return best_model, self_max_test_p, self_max_test_r, self_max_test_f1


print('self iteration {}'.format(opt.self_iteration))

# targets_zh = ["entertainment.txt", "home.txt", "education.txt", "finance.txt"]
targets_zh = ["topic1.txt", "topic2.txt"]

for domain in targets_zh:
    print('\n############ target domain {}, self_strategy {} ############\n'.format(domain,opt.self_strategy))
    
    start_time = time.time()
    # GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # init model
    model = DrlClassifier(opt)
    model.to(device)
    content_discriminator_params, \
    emotion_discriminator_params, cause_discriminator_params, \
    ec_discriminator_params, ce_discriminator_params, vae_and_classifier_params = model.get_params()

    # define optimizers
    content_disc_opt = torch.optim.RMSprop(content_discriminator_params, lr=opt.adv_lr)
    emotion_disc_opt = torch.optim.RMSprop(emotion_discriminator_params, lr=opt.adv_lr)
    cause_disc_opt = torch.optim.RMSprop(cause_discriminator_params, lr=opt.adv_lr)
    ec_disc_opt = torch.optim.RMSprop(ec_discriminator_params, lr=opt.adv_lr)
    ce_disc_opt = torch.optim.RMSprop(ce_discriminator_params, lr=opt.adv_lr)
    vae_and_cls_opt = torch.optim.Adam(vae_and_classifier_params, lr=opt.vae_lr)
    opts = [content_disc_opt, emotion_disc_opt, cause_disc_opt, ec_disc_opt, ce_disc_opt, vae_and_cls_opt]

    # load data
    train_path = "domains/Englishnovel_multiple/topic0.txt"
    test_path = "domains/Englishnovel_multiple/" + domain
    # test_path = "pair_data/emotion/" + domain
    # train_df, _, _ = read_ECPE_data(train_path)
    # test_df, test_docs_pair_size, num_unpred_pairs = read_ECPE_data(test_path, test=True)
    train_df, _ = read_ECPE_data(train_path)
    test_df, test_docs_pair_size = read_ECPE_data(test_path, test=True)
    train_dataset = ECPEDataset(train_df)
    test_dataset = ECPEDataset(test_df)
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=opt.batch_size,
                                                    shuffle=True,
                                                    num_workers=0
                                                    )
    test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=test_df.shape[0],
                                                   shuffle=False,
                                                   num_workers=0
                                                   )

    best_trained_model = train(train_data_loader, test_data_loader, model, opts, device, num_unpred_pairs=0)

    # self-training
    print("############ Self-training Start ############")
    self_best_trained_model = best_trained_model
    self_metrics = [0.0, 0.0, 0.0]
    for i in range(opt.self_iteration):
        print("############ Iteration {} ############".format(i + 1))
        self_train_df = generate_self_train_data(test_docs_pair_size, test_df, test_data_loader,
                                                 self_best_trained_model,
                                                 strategy=opt.self_strategy)
        self_train_dataset = ECPEDataset(self_train_df)
        self_train_data_loader = torch.utils.data.DataLoader(self_train_dataset,
                                                             batch_size=opt.batch_size,
                                                             shuffle=True,
                                                             num_workers=0
                                                             )
        self_best_trained_model, self_p, self_r, self_f1 = train(self_train_data_loader,
                                                                 test_data_loader,
                                                                 self_best_trained_model,
                                                                 opts, device, num_unpred_pairs=0,
                                                                 self_metrics=self_metrics,
                                                                 self_train=True)
        self_metrics[0] = self_p
        self_metrics[1] = self_r
        self_metrics[2] = self_f1

    print("---running time: %s minutes ---" % ((time.time() - start_time) / 60))
