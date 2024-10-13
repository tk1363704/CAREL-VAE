"""
Written by Yujin Huang(Jinx)
Started 28/12/2021 10:16 pm
Last Editted 
final version of causal discovery classifier
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
parser.add_argument('--source_domain', default='society_num', help='zh-society_num, en-history_num')
parser.add_argument('--target_domain', default='education',
                    help='zh-finance, education, entertainment and home, en-war, romance, adventure and biography')
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
opt.model_id = str(uuid4())

if opt.language == 'zh':
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext',
                                              cache_dir="model/hfl_chinese-roberta-wwm-ext")
    bow = get_bow_zh(opt.bow_file)
elif opt.language == 'en':
    opt.bow_file = 'data/all_data_pair_en.txt'
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base',
                                                 cache_dir="model/roberta-base")
    bow = get_bow_en(opt.bow_file)

opt.pair_bow_dim = len(bow)

timestr = time.strftime("%Y%m%d-%H%M%S")
if opt.language == 'zh':
    log = open('zh_final_cause_pair_extraction_mul_log_' + timestr + '.txt', 'w', buffering=1)
elif opt.language == 'en':
    log = open('en_final_cause_pair_extraction_mul_log_' + timestr + '.txt', 'w', buffering=1)

sys.stdout = log
sys.stderr = log
print(''.join(f'{k}={v}\n' for k, v in vars(opt).items()))


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


class MMDStatistic:
    def __init__(self, n_1, n_2):
        self.n_1 = n_1
        self.n_2 = n_2

        # The three constants used in the test.
        self.a00 = 1. / (n_1 * (n_1 - 1))
        self.a11 = 1. / (n_2 * (n_2 - 1))
        self.a01 = - 1. / (n_1 * n_2)

    def __call__(self, sample_1, sample_2, alphas, ret_matrix=False):
        sample_12 = torch.cat((sample_1, sample_2), 0)
        distances = pdist(sample_12, sample_12, norm=2)

        kernels = None
        for alpha in alphas:
            kernels_a = torch.exp(- alpha * distances ** 2)
            if kernels is None:
                kernels = kernels_a
            else:
                kernels = kernels + kernels_a

        k_1 = kernels[:self.n_1, :self.n_1]
        k_2 = kernels[self.n_1:, self.n_1:]
        k_12 = kernels[:self.n_1, self.n_1:]

        mmd = (2 * self.a01 * k_12.sum() +
               self.a00 * (k_1.sum() - torch.trace(k_1)) +
               self.a11 * (k_2.sum() - torch.trace(k_2)))
        if ret_matrix:
            return mmd, kernels
        else:
            return mmd

    def pval(self, distances, n_permutations=1000):
        if isinstance(distances, Variable):
            distances = distances.data
        return permutation_test_mat(distances.cpu().numpy(),
                                    self.n_1, self.n_2,
                                    n_permutations,
                                    a00=self.a00, a11=self.a11, a01=self.a01)


def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1 ** 2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2 ** 2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)


def permutation_test_mat(*args, **kwargs):  # real signature unknown
    pass


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

    ckpt_path = os.path.join(ckpt_path, opt.model_id + ".pt")
    torch.save(state, ckpt_path)


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
            predicted_df["label"] = final_outputs
            predicted_df["label"] = predicted_df["label"].apply(lambda x: x[0])

    df = pd.DataFrame(columns=["pair", "label", "emotion"])
    curr_index = 0

    for doc_pair_size in test_docs_pair_size:
        # positive pair: (emotion, cause); negative pair: (emotion, non-cause)
        max_pos_prob = float("-inf")
        max_neg_prob = float("-inf")
        pos_pair = None
        pos_emotion = None
        neg_pair = None
        neg_emotion = None

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
                elif 0.5 >= prob > max_neg_prob:
                    neg_pair = pair_label_data["pair"]
                    max_neg_prob = prob
            # strategy 2: highest score as positive and random negative
            elif strategy == "random":
                prob_dict[index] = prob
                sort_prob_dict = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
                pos_pair = predicted_df.iloc[sort_prob_dict[0][0]]["pair"]
                pos_emotion = predicted_df.iloc[sort_prob_dict[0][0]]["emotion"]
                if len(sort_prob_dict) == 1: continue
                neg_ran_index = randint(1, len(sort_prob_dict) - 1)
                neg_pair = predicted_df.iloc[sort_prob_dict[neg_ran_index][0]]["pair"]
                neg_emotion = predicted_df.iloc[sort_prob_dict[neg_ran_index][0]]["emotion"]
            elif strategy == "extreme":
                prob_dict[index] = prob
                sort_prob_dict = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
                pos_pair = predicted_df.iloc[sort_prob_dict[0][0]]["pair"]
                neg_pair = predicted_df.iloc[sort_prob_dict[-1][0]]["pair"]

        curr_index += doc_pair_size

        if pos_pair is not None and neg_pair is not None:
            df = df.append({'pair': pos_pair, 'label': 1, 'emotion': pos_emotion}, ignore_index=True)
            df = df.append({'pair': neg_pair, 'label': 0, 'emotion': neg_emotion}, ignore_index=True)

    return df


def train(train_loader, test_loader, model, optimizers, device, num_unpred_pairs, self_metrics=None, self_train=False):
    # obtain all optimizers
    vae_and_cls_opt = optimizers[0]

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
        running_loss = 0

        model.train()
        print('\n############ Epoch {}: Training Start ############\n'.format(epoch))
        for iteration, batch in enumerate(train_loader):
            ids = batch['input_ids'].to(device, dtype=torch.long)
            att_masks = batch['attention_masks'].to(device, dtype=torch.long)
            token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
            labels = batch['labels'].to(device, dtype=torch.float)
            emo_labels = batch['emo_labels'].to(device, dtype=torch.long)
            cau_labels = batch['cau_labels'].to(device, dtype=torch.float)
            bow_reps = batch['bow_reps'].to(device, dtype=torch.float)

            vae_and_cls_loss = model(ids, att_masks, token_type_ids, emo_labels,
                                     cau_labels,
                                     labels, bow_reps,
                                     iteration)

            with torch.autograd.set_detect_anomaly(True):

                # update VAE and classifier parameters
                vae_and_cls_opt.zero_grad()
                vae_and_cls_loss.backward()
                vae_and_cls_opt.step()

            # running loss for monitoring
            running_loss += vae_and_cls_loss.item()

            # print running loss every 10 mini-batches
            if iteration % 10 == 9:
                print('[%d, %5d] training loss: %.4f' %
                      (epoch, iteration + 1, running_loss / 10))
                running_loss = 0.0

        model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(test_loader):
                ids = batch['input_ids'].to(device, dtype=torch.long)
                att_masks = batch['attention_masks'].to(device, dtype=torch.long)
                token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
                labels = batch['labels'].to(device, dtype=torch.float)

                labels = labels.cpu().detach().numpy().tolist()
                preds = model.get_pair_preds(ids, att_masks, token_type_ids)
                # append the number of unpredicted pairs
                labels += [[1]] * num_unpred_pairs
                preds += [[0]] * num_unpred_pairs

                # precision, recall, and f1 score
                p = precision_score(labels, preds, average="binary")
                r = recall_score(labels, preds, average="binary")
                f1 = f1_score(labels, preds, average="binary")
                print("current test cause precision: {:.4f}, recall: {:.4f}, f1 socre: {:.4f}".format(p, r, f1))
                print("current test pair precision: {:.4f}, recall: {:.4f}, f1 socre: {:.4f}\n".format(p, r, f1))

                # create checkpoint variable and add important data
                checkpoint = model.state_dict()

                # save the model if test f1 score has increased (source domain)
                if f1 > max_test_f1 and not self_train:
                    print('test f1 score increased ({:.4f} --> {:.4f}).  Saving model...\n'.format(max_test_f1, f1))
                    # save checkpoint as best model
                    save_ckp(checkpoint, opt.best_model_path)

                    max_test_p = p
                    max_test_r = r
                    max_test_f1 = f1

                elif f1 > self_max_test_f1 and self_train:
                    print(
                        'test f1 score increased ({:.4f} --> {:.4f}).  Saving model...\n'.format(self_max_test_f1, f1))

                    save_ckp(checkpoint, opt.best_model_path)

                    self_max_test_p = p
                    self_max_test_r = r
                    self_max_test_f1 = f1

                if not self_train:
                    print(
                        "max test cause precision: {:.4f}, recall: {:.4f}, f1 socre: {:.4f}".format(max_test_p,
                                                                                                    max_test_r,
                                                                                                    max_test_f1))
                    print(
                        "max test pair precision: {:.4f}, recall: {:.4f}, f1 socre: {:.4f}\n".format(max_test_p,
                                                                                                     max_test_r,
                                                                                                     max_test_f1))
                else:
                    print("self max test cause precision: {:.4f}, recall: {:.4f}, f1 socre: {:.4f}".format(
                        self_max_test_p,
                        self_max_test_r,
                        self_max_test_f1))
                    print("self max test pair precision: {:.4f}, recall: {:.4f}, f1 socre: {:.4f}\n".format(
                        self_max_test_p,
                        self_max_test_r,
                        self_max_test_f1))

    # obtain the best trained model
    best_model = load_ckp(os.path.join(opt.best_model_path, opt.model_id + ".pt"), model)

    if not self_train:
        return best_model
    else:
        return best_model, self_max_test_p, self_max_test_r, self_max_test_f1


print('\n############ target domain {} ############\n'.format(opt.target_domain))
start_time = time.time()
# GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# init model
model = DrlClassifier(opt)
model.to(device)
vae_and_classifier_params = model.get_params()

# define optimizers
vae_and_cls_opt = torch.optim.Adam(vae_and_classifier_params, lr=opt.vae_lr)
opts = [vae_and_cls_opt]

# language
if opt.language == 'zh':
    dir = "domains/THUCTC_multiple/"
elif opt.language == 'en':
    dir = "domains/Englishnovel_multiple/"

# load data
train_path = dir + opt.source_domain + ".txt"
test_path = "pair_data/emotion/" + opt.target_domain + ".txt"
train_df, _, _ = read_ECPE_data(train_path)
test_df, test_docs_pair_size, num_unpred_pairs = read_ECPE_data(test_path, test=True)
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

best_trained_model = train(train_data_loader, test_data_loader, model, opts, device, num_unpred_pairs)

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
                                                             opts, device, num_unpred_pairs,
                                                             self_metrics=self_metrics,
                                                             self_train=True)
    self_metrics[0] = self_p
    self_metrics[1] = self_r
    self_metrics[2] = self_f1

print("---running time: %s minutes ---" % ((time.time() - start_time) / 60))
