'''
Based on chinese-roberta-wwm-ext-base, we fine-tuned an NLI version on 4 Chinese Natural Language Inference (NLI) datasets, with totaling 1,014,787 samples.
The predicted_label corresponds to the predicted entailment relationship between the premise and hypothesis, where 0 means contradiction, 1 means neutral, and 2 means entailment.
'''
import json

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer
import torch
from torch import cuda
from constant import *


def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


# device = torch.device("mps")
tokenizer = T5Tokenizer.from_pretrained("ClueAI/ChatYuan-large-v1")
pretrained_model_path = PROJECT_ABSOLUTE_PATH + "/chatyuan_model/model_files/"
model_trained = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_path)
device = 'cuda' if cuda.is_available() else 'cpu'
model_trained.to(device)
print("Basic configuration end...")

def preprocess(text):
    return text.replace("\n", "_")


def postprocess(text):
    return text.replace("_", "\n")


def calc_prob(input_, target_, with_input_prob=False, with_length_normalization=False, with_logsum=False):
    eos_id = tokenizer.encode("</s>", return_tensors='pt').to(device)
    input_ids = tokenizer.encode(input_, return_tensors='pt').to(device)
    input_id_list = input_ids.tolist()[0]
    target_ids = tokenizer.encode(target_, return_tensors='pt').to(device)
    target_id_list = target_ids.tolist()[0]
    length_normalization, e_i_decode_prob = 1.0, 1.0

    if with_input_prob is True:
        e_i_logits = model_trained(input_ids=eos_id, decoder_input_ids=input_ids).logits
        e_i_probs = torch.softmax(e_i_logits, dim=1)
        e_i_prob_list_ = e_i_probs.tolist()[0]
        # e_i_decode_prob = p(input|eos)
        for i, id_ in enumerate(input_id_list):
            e_i_decode_prob *= e_i_prob_list_[i][id_]

    if with_length_normalization is True:
        length_normalization = 0.0 if len(target_id_list) == 0 else 1.0 / len(target_id_list)

    i_t_logits = model_trained(input_ids=input_ids, decoder_input_ids=target_ids).logits
    i_t_probs = torch.softmax(i_t_logits, dim=1)
    i_t_prob_list_ = i_t_probs.tolist()[0]
    # i_t_decode_prob_ = p(target|input)
    i_t_decode_prob_ = 1.0 if with_logsum is False else 0.0
    if with_logsum is False:
        for i, id_ in enumerate(target_id_list):
            i_t_decode_prob_ *= i_t_prob_list_[i][id_]
    else:
        for i, id_ in enumerate(target_id_list):
            i_t_decode_prob_ += torch.log(torch.tensor(i_t_prob_list_[i][id_])).item()

    return i_t_decode_prob_ * e_i_decode_prob * length_normalization


def answer_fn(text, sample=False, top_p=0.6):
    '''sample：是否抽样。生成任务，可以设置为True;
       top_p：0-1之间，生成的内容越多样;
    '''
    text = preprocess(text)
    encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=768, return_tensors="pt").to(device)
    if not sample:  # 不进行采样
        out = model_trained.generate(**encoding, return_dict_in_generate=True, output_scores=True, max_length=128,
                                     num_beams=4, length_penalty=0.6)
    else:  # 采样（生成）
        out = model_trained.generate(**encoding, return_dict_in_generate=True, output_scores=True, max_length=128,
                                     do_sample=True, top_p=top_p)

    yes_prob = 0.0
    first_output_ids = out["scores"][0].argmax(dim=1).tolist()
    if first_output_ids[0] == 12:
        second_output_probs = torch.softmax(out["scores"][1], dim=1).tolist()
        yes_prob = second_output_probs[0][15]
    else:
        print('error, the first output ids is: {}'.format(first_output_ids))

    out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
    return postprocess(out_text[0]), yes_prob

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
        emotion_index_int, event_index_int = int(pair[0]), int(pair[1])
        if event_index_int <= emotion_index_int:
            temporal_positive_count += 1

        emotion_sentence, event_sentence = dict_[emotion_index], dict_[event_index]

        # todo: event and emotion is in the same sentence

        # event_to_emotion_text = "假设“{}”我们可以推断“{}”？是的,不是,或也许？_答案：".format(event_sentence, emotion_sentence)
        # event_to_emotion_result, event_to_emotion_yes_prob = answer_fn(event_to_emotion_text, sample=False, top_p=0.6)
        #
        # emotion_to_event_text = "假设“{}”我们可以推断“{}”？是的,不是,或也许？_答案：".format(emotion_sentence, event_sentence)
        # emotion_to_event_result, emotion_to_event_yes_prob = answer_fn(emotion_to_event_text, sample=False, top_p=0.6)

        # count_ += 1
        # if count_ % 10 == 0:
        #     print(count_)
        # if event_to_emotion_yes_prob >= emotion_to_event_yes_prob:
        #     positive_count += 1

        event_to_emotion_prob = calc_prob(event_sentence, emotion_sentence, with_input_prob=False, with_length_normalization=True, with_logsum=True)
        emotion_to_event_prob = calc_prob(emotion_sentence, event_sentence, with_input_prob=False, with_length_normalization=True, with_logsum=True)

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
