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
train_batch_size = 16
# num_epochs = 9 # emotion
num_epochs = 8 # cause
model_save_path = "ECPE_model/emo_domain_sentence_transformer_english"

pre_trained_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', cache_folder="model")

# domains = ["history.txt", "war.txt", "romance.txt", "adventure.txt", "biography.txt", "topic5.txt", "topic6.txt", "topic7.txt"]
domains = ["history.txt", "war.txt", "romance.txt", "adventure.txt", "biography.txt"]

df = pd.DataFrame(columns=["sen", "label"])

for d in domains:
    data_file = open("domains/Englishnovel_multiple/" + d, encoding="utf8")

    while True:
        line = data_file.readline()
        if not line:
            break
        if re.search("[0-9]{1,4}[\s][0-9]{1,2}", line):
            doc_len = int(line.strip().split(" ")[1])
            # pair info
            pos_pairs = eval('[' + data_file.readline().strip() + ']')
            emo_list, cau_list = zip(*pos_pairs)
            pos_pairs = [(emo_list[i], cau_list[i]) for i in range(len(emo_list))]
            true_emotions = list([e[0] for e in pos_pairs])
            true_causes = list([c[1] for c in pos_pairs])

            # doc's sentences
            for i in range(doc_len):
                sentence = data_file.readline().split(',')[3]

                if (i + 1) in true_emotions:
                # if (i + 1) in true_causes:
                    if d == "history.txt":
                        label = 0
                    elif d == "war.txt":
                        label = 0
                    elif d == "romance.txt":
                        label = 0
                    elif d == "adventure.txt":
                        label = 0
                    elif d == "biography.txt":
                        label = 0
                    elif d == "topic5.txt":
                        label = 5
                    elif d == "topic6.txt":
                        label = 6
                    elif d == "topic7.txt":
                        label = 7

                    df = df.append({'sen': sentence, 'label': label}, ignore_index=True)

train_samples = []

for ind in df.index:
    input_example = InputExample(texts=[df['sen'][ind]], label=df['label'][ind])
    train_samples.append(input_example)

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
# train_loss = losses.BatchSemiHardTripletLoss(model=pre_trained_model, margin=4.5) # emotion vis 4.5
train_loss = losses.BatchSemiHardTripletLoss(model=pre_trained_model, margin=5) # cause vis margin=4.55 4.35

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
pre_trained_model.fit(train_objectives=[(train_dataloader, train_loss)],
                      epochs=num_epochs,
                      warmup_steps=warmup_steps,
                      output_path=model_save_path)
