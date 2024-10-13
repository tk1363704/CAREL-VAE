"""
Written by Yujin Huang(Jinx)
Started 15/09/2021 2:10 pm
Last Editted 

Description of the purpose of the code
"""
from sentence_transformers import SentenceTransformer, models
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import os


def load(path, fine_tune=True):
    # check whether
    use_cuda = torch.cuda.is_available()

    training_file = "cause_train.npy"
    validation_file = "cause_validation.npy"
    test_file = "cause_test.npy"

    train = np.load(os.path.join(path, training_file))
    validation = np.load(os.path.join(path, validation_file))
    test = np.load(os.path.join(path, test_file))

    X_train, y_train, = train[:, 0].tolist(), train[:, 1].astype(int)
    X_valid, y_valid = validation[:, 0].tolist(), validation[:, 1].astype(int)
    X_test, y_test = test[:, 0].tolist(), test[:, 1].astype(int)

    # chinese sentence transformer
    if fine_tune:
        # fine-tuned
        embedder = SentenceTransformer('../ECPE_model/fine_tuned_sentence_transformer_chinese')
    else:
        # pre-trained
        embedder = SentenceTransformer('cyclone/simcse-chinese-roberta-wwm-ext', cache_folder="model")

    X_train_embeddings = embedder.encode(X_train)
    X_valid_embeddings = embedder.encode(X_valid)
    X_test_embeddings = embedder.encode(X_test)

    X_train, y_train, = torch.tensor(X_train_embeddings, dtype=torch.float32), torch.tensor(y_train, dtype=torch.int)
    X_valid, y_valid = torch.tensor(X_valid_embeddings, dtype=torch.float32), torch.tensor(y_valid, dtype=torch.int)
    X_test, y_test = torch.tensor(X_test_embeddings, dtype=torch.float32), torch.tensor(y_test, dtype=torch.int)

    if use_cuda:
        X_train, X_valid, X_test, y_train, y_valid, y_test = X_train.cuda(), X_valid.cuda(), X_test.cuda(), y_train.cuda(), y_valid.cuda(), y_test.cuda()

    return X_train, X_valid, X_test, y_train, y_valid, y_test
