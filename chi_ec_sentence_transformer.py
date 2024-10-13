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
# num_epochs = 8 # emotion
num_epochs = 7 # cause
model_save_path = "ECPE_model/cau_domain_sentence_transformer_chinese"

pre_trained_model = SentenceTransformer('cyclone/simcse-chinese-roberta-wwm-ext', cache_folder="model")
# pre_trained_model = SentenceTransformer('./ECPE_model/ec_domain_sentence_transformer_chinese')

domains = ["society.txt", "entertainment.txt", "home.txt", "education.txt", "finance.txt", "game.txt", "technology.txt"]
# domains = ["society.txt", "entertainment.txt", "home.txt", "education.txt", "finance.txt"]
# domains = ["society.txt", "entertainment.txt"]
df = pd.DataFrame(columns=["sen", "label"])

for d in domains:
    data_file = open("domains/THUCTC_multiple/" + d, encoding="utf8")

    while True:
        line = data_file.readline()
        if not line:
            break
        if re.search("[0-9]{1,4}[\s][0-9]{1,2}", line):
            doc_len = int(line.strip().split(" ")[1])
            # pair info
            pos_pairs = data_file.readline().strip().split(", ")
            pos_pairs = [eval(x) for x in pos_pairs]
            true_emotions = list([e[0] for e in pos_pairs])
            true_causes = list([e[1] for e in pos_pairs])

            # doc's sentences
            for i in range(doc_len):
                sentence = data_file.readline().split(',')[3].replace(" ", "")

                # if (i + 1) in true_emotions:
                if (i + 1) in true_causes:
                    if d == "society.txt":
                        label = 0
                    elif d == "entertainment.txt":
                        label = 1
                    elif d == "home.txt":
                        label = 2
                    elif d == "education.txt":
                        label = 3
                    elif d == "finance.txt":
                        label = 4
                    elif d == "game.txt":
                        label = 5
                    elif d == "technology.txt":
                        label = 6
                    elif d == "politics.txt":
                        label = 7

                    df = df.append({'sen': sentence, 'label': label}, ignore_index=True)

train_samples = []

for ind in df.index:
    input_example = InputExample(texts=[df['sen'][ind]], label=df['label'][ind])
    train_samples.append(input_example)

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
# train_loss = losses.BatchSemiHardTripletLoss(model=pre_trained_model, margin=400) # emotion vis
train_loss = losses.BatchSemiHardTripletLoss(model=pre_trained_model, margin=4.45) # cause vis margin=4.55 4.35

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
pre_trained_model.fit(train_objectives=[(train_dataloader, train_loss)],
                      epochs=num_epochs,
                      warmup_steps=warmup_steps,
                      output_path=model_save_path)
