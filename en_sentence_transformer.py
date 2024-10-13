"""
Written by Yujin Huang(Jinx)
Started 28/09/2021 3:50 pm
Last Editted 

Description of the purpose of the code
"""
import numpy as np
import math
import logging
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd
import re

# hyperparameter setting
train_batch_size = 32
num_epochs = 5
model_save_path = "ECPE_model/doc_domain_sentence_transformer_english"

pre_trained_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', cache_folder="model")
# pre_trained_model.max_seq_length=25

# domains = ["history.txt", "war.txt", "romance.txt", "adventure.txt", "biography.txt", "topic5.txt", "topic6.txt", "topic7.txt"]
# df = pd.DataFrame(columns=["doc", "label"])

# for d in domains:
#     data_file = open("domains/Englishnovel_multiple/" + d, encoding="utf8")

#     while True:
#         line = data_file.readline()
#         if not line:
#             break
#         if re.search("[0-9]{1,4}[\s][0-9]{1,2}", line):
#             doc_len = int(line.strip().split(" ")[1])
#             # pair info
#             data_file.readline().strip().split(", ")
#             content = ''
#             # doc's sentences
#             for i in range(doc_len):
#                 sentence = data_file.readline().split(',')[3]
#                 content += sentence

#             if d == "history.txt":
#                 label = 0
#             elif d == "war.txt":
#                 label = 1
#             elif d == "romance.txt":
#                 label = 2
#             elif d == "adventure.txt":
#                 label = 3
#             elif d == "biography.txt":
#                 label = 4
#             elif d == "topic5.txt":
#                 label = 5
#             elif d == "topic6.txt":
#                 label = 6
#             elif d == "topic7.txt":
#                 label = 6

#             df = df.append({'doc': content, 'label': label}, ignore_index=True)

df = pd.read_csv('en_vis_train.csv')
df['topic'] = df['topic'].replace(['History','War','Romance', 'Adventure', 'Biography','5', '6', '7'],[0,1,2,3,4,5,6,7])
# df['topic'] = df['topic'].replace(['History','War','Romance', 'Adventure', 'Biography'],[0,1,2,3,4])

train_samples = []

for ind in df.index:
    input_example = InputExample(texts=[df['title'][ind]], label=df['topic'][ind])
    train_samples.append(input_example)

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.BatchSemiHardTripletLoss(model=pre_trained_model, margin=6) #candidate 

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
pre_trained_model.fit(train_objectives=[(train_dataloader, train_loss)],
                      epochs=num_epochs,
                      warmup_steps=warmup_steps,
                      output_path=model_save_path)
