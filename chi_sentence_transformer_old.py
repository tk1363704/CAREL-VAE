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

np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

# hyperparameter setting
train_batch_size = 16
num_epochs = 4
model_save_path = "ECPE_model/fine_tuned_sentence_transformer_chinese"

pre_trained_model = SentenceTransformer('cyclone/simcse-chinese-roberta-wwm-ext', cache_folder="model")

cau_corpus = np.load("data/ECPE_cause/cau_corpus.npy")

train_samples = []

for c in cau_corpus:
    input_example = InputExample(texts=[c[0]], label=c[1])
    train_samples.append(input_example)

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.BatchAllTripletLoss(model=pre_trained_model, margin=10)

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
pre_trained_model.fit(train_objectives=[(train_dataloader, train_loss)],
                      epochs=num_epochs,
                      warmup_steps=warmup_steps,
                      output_path=model_save_path)
