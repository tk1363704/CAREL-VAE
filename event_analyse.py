import json
import os
import jieba.posseg as pseg
import nltk
import stanfordnlp
import thulac
from snownlp import SnowNLP

nltk.download('punkt')
# Load a THULAC tagger
thu_tagger = thulac.thulac()
# # Load a StanfordNLP tagger
# stanfordnlp.download('zh')


PROJECT_ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


def get_event_sentences(path):
    event_sentences = []
    with open(path, 'r') as f:
        cont_ = json.load(f)
    for _, value in cont_.items():
        dialogue = value["content"]
        c_e_pairs = dialogue[0].split(', ')
        pairs = []
        for c_e_pair in c_e_pairs:
            pair = c_e_pair.strip().replace('(', '').replace(')', '').split(',')
            pairs.append(pair)
        event_ids = [int(x[1]) for x in pairs]
        event_sentences.extend([dialogue[x] for x in event_ids])
    return event_sentences


def find_verbs(sentence, tool):
    utterance = sentence.split(',')[-1].replace(' ', '').strip()
    verb_flag = False

    if tool == 'jieba':
        # Use jieba to segment the sentence and identify the part of speech of each word
        words = pseg.cut(utterance)
        # Loop through each word in the sentence and print the verb words
        verb_flag = False
        for word, flag in words:
            if flag == 'v':
                print(word)
                verb_flag = True
                break
    elif tool == "thulac":
        # Perform POS analysis on the words using THULAC
        pos_tags = thu_tagger.cut(utterance, text=True)
        # Loop through each word and print the verb words
        verb_flag = False
        for word_tag in pos_tags.split(' '):
            word, tag = word_tag.split('_')[0], word_tag.split('_')[1]
            if tag == 'v':
                print(word)
                verb_flag = True
                break
    elif tool == "stanford":
        # Load the Chinese models
        nlp = stanfordnlp.Pipeline(lang='zh', processors='pos')
        doc = nlp(utterance)
        # Loop through each word and print the verb words
        verb_flag = False
        # Obtain the POS tags for each token in the sentence
        for token in doc.sentences[0].tokens:
            print(token.text, token.pos)
    elif tool == "snow":
        # Perform POS analysis on the words using SnowNLP
        s = SnowNLP(utterance)
        pos_tags = s.tags
        # Loop through each word and print the verb words
        verb_flag = False
        for word, tag in pos_tags:
            if tag == 'v':
                print(word)
                verb_flag = True
                break
    return verb_flag


def main():
    path = PROJECT_ABSOLUTE_PATH+'/data/new.json'
    event_sentences = get_event_sentences(path)
    verb_count = 0
    for event_sentence in event_sentences:
        if find_verbs(event_sentence, 'snow'):
            verb_count += 1
    print('We have {} event sentences in total.'.format(len(event_sentences)))
    print('The verb proportion is {}'.format(float(verb_count/len(event_sentences))))



if __name__ == "__main__":
    main()