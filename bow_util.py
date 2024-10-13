"""
Written by Yujin Huang(Jinx)
Started 4/01/2022 7:59 pm
Last Editted 

Description of the purpose of the code
"""
import re
import jieba
from sklearn.feature_extraction.text import CountVectorizer


def tokenize_zh(text):
    filter_zh = re.compile(u'[^\u4E00-\u9FA5]')  # non-Chinese unicode range
    text = filter_zh.sub(r'', text)  # remove all non-Chinese characters
    words = jieba.lcut(text)
    return words


def get_bow_zh(file_path):
    data_file = open(file_path, encoding="utf8")
    corpus = []

    while True:
        line = data_file.readline()
        if not line:
            break
        if re.search("[0-9]{1,4}[\s][0-9]{1,2}", line):
            doc_len = int(line.strip().split(" ")[1])
            # skip pair info
            data_file.readline()
            # add a doc's sentences to the corpus
            for i in range(doc_len):
                sentence = data_file.readline().strip().split(",")[3].replace(" ", "")
                corpus.append(sentence)

    vectorizer = CountVectorizer(tokenizer=tokenize_zh)
    vectorizer.fit_transform(corpus)
    bow_features = vectorizer.get_feature_names()
    return bow_features

def bow_tokenize(sentence, tokenizer=None):
    sentence = sentence.lower()  # convert text to lowercase
    sentence = re.sub(r'[^\w\s]', '', sentence)
    # tokens = tokenizer.tokenize(sentence)
    tokens = sentence.split(" ")
    tokens_without_space_markers = [x for x in [token.replace('Ġ', '') for token in tokens] if x != '']
    return tokens_without_space_markers

def get_bow_en(file_path, bow_optimize=False, tokenizer=None):
    data_file = open(file_path, encoding="utf8")
    # corpus = []
    if bow_optimize is False:
        corpus = []
    else:
        corpus = set()
        corpus.add('sep')

    while True:
        line = data_file.readline()
        if not line:
            break
        if re.search("[0-9]{1,4}[\s][0-9]{1,2}", line):
            doc_len = int(line.strip().split(" ")[1])
            # skip pair info
            data_file.readline()
            # add a doc's sentences to the corpus
            for i in range(doc_len):
                line = data_file.readline().strip().split(",")[3]
                if bow_optimize is False:
                    sentence = line.replace(" ", "")
                    corpus.append(sentence)
                else:
                    tokens = bow_tokenize(line, tokenizer)
                    for token in tokens:
                        corpus.add(token)

    vectorizer = CountVectorizer()
    vectorizer.fit_transform(corpus)
    bow_features = vectorizer.get_feature_names()
    return bow_features
