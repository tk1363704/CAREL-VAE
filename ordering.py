'''
Based on chinese-roberta-wwm-ext-base, we fine-tuned an NLI version on 4 Chinese Natural Language Inference (NLI) datasets, with totaling 1,014,787 samples.
The predicted_label corresponds to the predicted entailment relationship between the premise and hypothesis, where 0 means contradiction, 1 means neutral, and 2 means entailment.
'''
import json

from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import torch
from constant import *


def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


# device = torch.device("mps")
tokenizer = BertTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-330M-NLI')
model = BertForSequenceClassification.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-330M-NLI')
new_dialogues = read_json(PROJECT_ABSOLUTE_PATH + '/data/new.json')
count_ = 0
positive_count = 0
temporal_positive_count = 0
for _, dialogue in new_dialogues.items():
    content = dialogue['content']
    dict_ = {}

    for line in content[1:]:
        info_ = line.strip().split(',')
        dict_[info_[0]] = info_[-1].replace(' ', '')

    entailments = content[0].split(', ')
    for entailment in entailments:
        pair = entailment.strip().replace('(', '').replace(')', '').split(',')
        emotion_index, event_index = pair[0], pair[1]
        if event_index <= emotion_index:
            temporal_positive_count += 1

        emotion_sentence, event_sentence = dict_[emotion_index], dict_[event_index]
        output = model(torch.tensor([tokenizer.encode(event_sentence, emotion_sentence)]))
        event_to_emotion_prob = torch.nn.functional.softmax(output.logits, dim=-1).tolist()[0][2]
        output = model(torch.tensor([tokenizer.encode(emotion_sentence, event_sentence)]))
        emotion_to_event_prob = torch.nn.functional.softmax(output.logits, dim=-1).tolist()[0][2]
        count_ += 1
        if count_ % 10 == 0:
            print(count_)
        if event_to_emotion_prob >= emotion_to_event_prob:
            positive_count += 1

print('total event-emotion pairs: {0}, total positive pairs: {1}, positive probability is {2}'.format(count_, positive_count, float(positive_count/count_)))
print('total event-emotion pairs: {0}, total temporal positive pairs: {1}, positive probability is {2}'.format(count_, temporal_positive_count, float(temporal_positive_count/count_)))



# premise = '男人俩互相交换一个很大的金属环，骑着火车向相反的方向行驶'
# hypothesis = '婚礼正在教堂举行'
#
# output = model(torch.tensor([tokenizer.encode(premise, hypothesis)]))
# prob = torch.nn.functional.softmax(output.logits, dim=-1)
# lst = prob.tolist()
# print('entailment prob = {}'.format(lst[0][2]))
