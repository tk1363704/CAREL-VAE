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
import re


def load(fine_tune=True):
    # check whether
    use_cuda = torch.cuda.is_available()

    # training_file = "cause_train.npy"
    # validation_file = "cause_validation.npy"
    # test_file = "cause_test.npy"
    #
    # train = np.load(os.path.join(path, training_file))
    # validation = np.load(os.path.join(path, validation_file))
    # test = np.load(os.path.join(path, test_file))
    #
    # X_train, y_train, = train[:, 0].tolist(), train[:, 1].astype(int)
    # X_valid, y_valid = validation[:, 0].tolist(), validation[:, 1].astype(int)
    # X_test, y_test = test[:, 0].tolist(), test[:, 1].astype(int)

    # causes for happiness, sadness, disgust, surprise, fear, anger
    s_cau_happiness, s_cau_sadness, s_cau_disgust, s_cau_surprise, s_cau_fear, s_cau_anger, s_cau_none = ([] for i in
                                                                                                          range(7))
    t_cau_happiness, t_cau_sadness, t_cau_disgust, t_cau_surprise, t_cau_fear, t_cau_anger, t_cau_none = ([] for i in
                                                                                                          range(7))

    # source and target domains
    data_file = open("data/clause_keywords_emotion.txt", encoding="utf8")
    source = ["society.txt", "education.txt"]
    target = ["entertainment.txt", "home.txt"]

    # construct sentence embedding from source domain
    s_doc_ids = []
    t_doc_ids = []
    for s_file, t_file in zip(source, target):
        with open(os.path.join("data/category/", s_file), encoding="utf8") as infile:
            for sentence in infile:
                if re.search("[0-9]{1,4}[\s][0-9]{1,2}", sentence):
                    s_doc_ids.append(sentence.split(" ")[0])

        with open(os.path.join("data/category/", t_file), encoding="utf8") as infile:
            for sentence in infile:
                if re.search("[0-9]{1,4}[\s][0-9]{1,2}", sentence):
                    t_doc_ids.append(sentence.split(" ")[0])

    for sentence in data_file:
        # obtain the document id, emotion and clause
        id = sentence.split(",")[0]
        emotion = sentence.split(",")[1]
        clause = sentence.split(",")[-1].rstrip("\n").replace(" ", "")
        flag = sentence.split(",")[5]

        if id in s_doc_ids:
            # determine whether this clause is a cause of emotion
            if flag == "yes":
                if emotion == "happiness":
                    s_cau_happiness.append(clause)
                elif emotion == "sadness":
                    s_cau_sadness.append(clause)
                elif emotion == "disgust":
                    s_cau_disgust.append(clause)
                elif emotion == "surprise":
                    s_cau_surprise.append(clause)
                elif emotion == "fear":
                    s_cau_fear.append(clause)
                elif emotion == "anger":
                    s_cau_anger.append(clause)
            else:
                s_cau_none.append(clause)
        elif id in t_doc_ids:
            if flag == "yes":
                if emotion == "happiness":
                    t_cau_happiness.append(clause)
                elif emotion == "sadness":
                    t_cau_sadness.append(clause)
                elif emotion == "disgust":
                    t_cau_disgust.append(clause)
                elif emotion == "surprise":
                    t_cau_surprise.append(clause)
                elif emotion == "fear":
                    t_cau_fear.append(clause)
                elif emotion == "anger":
                    t_cau_anger.append(clause)
            else:
                t_cau_none.append(clause)

    # construct the corpus from source domain
    s_stat = {"cau_happiness": s_cau_happiness, "cau_sadness": s_cau_sadness, "cau_disgust": s_cau_disgust,
              "cau_surprise": s_cau_surprise, "cau_fear": s_cau_fear, "cau_anger": s_cau_anger, "cau_none": s_cau_none}

    t_stat = {"cau_happiness": t_cau_happiness, "cau_sadness": t_cau_sadness, "cau_disgust": t_cau_disgust,
              "cau_surprise": t_cau_surprise, "cau_fear": t_cau_fear, "cau_anger": t_cau_anger, "cau_none": t_cau_none}

    s_corpus = s_cau_happiness + s_cau_sadness + s_cau_disgust + s_cau_surprise + s_cau_fear + s_cau_anger + s_cau_none
    t_corpus = t_cau_happiness + t_cau_sadness + t_cau_disgust + t_cau_surprise + t_cau_fear + t_cau_anger + t_cau_none

    # assign labels to cause sentences
    s_cau_matrix = np.zeros((len(s_corpus), 2)).astype("object")
    count = 0

    for index, key in enumerate(s_stat):
        cau_vector = np.array([s_stat[key]]).T
        cau_matrix = np.zeros((len(s_stat[key]), 2)).astype("object")
        cau_matrix[:, :-1] = cau_vector
        cau_matrix[:, 1] = index
        row_index = len(s_stat[key]) + count
        s_cau_matrix[count:row_index, :] = cau_matrix
        count = row_index

    t_cau_matrix = np.zeros((len(t_corpus), 2)).astype("object")
    count = 0
    for index, key in enumerate(t_stat):
        cau_vector = np.array([t_stat[key]]).T
        cau_matrix = np.zeros((len(t_stat[key]), 2)).astype("object")
        cau_matrix[:, :-1] = cau_vector
        cau_matrix[:, 1] = index
        row_index = len(t_stat[key]) + count
        t_cau_matrix[count:row_index, :] = cau_matrix
        count = row_index

    X_train = s_cau_matrix[:, 0].tolist()
    y_train = s_cau_matrix[:, 1].astype(int)
    X_test = t_cau_matrix[:, 0].tolist()
    y_test = t_cau_matrix[:, 1].astype(int)

    # chinese sentence transformer
    if fine_tune:
        # fine-tuned
        embedder = SentenceTransformer('./ECPE_model/fine_tuned_sentence_transformer_chinese')
    else:
        # pre-trained
        embedder = SentenceTransformer('cyclone/simcse-chinese-roberta-wwm-ext', cache_folder="model")

    X_train_embeddings = embedder.encode(X_train)
    X_test_embeddings = embedder.encode(X_test)

    X_train, y_train, = torch.tensor(X_train_embeddings, dtype=torch.float32), torch.tensor(y_train, dtype=torch.int)
    X_test, y_test = torch.tensor(X_test_embeddings, dtype=torch.float32), torch.tensor(y_test, dtype=torch.int)

    if use_cuda:
        X_train, X_test, y_train, y_test = X_train.cuda(), X_test.cuda(), y_train.cuda(), y_test.cuda()

    return X_train, X_test, y_train, y_test
