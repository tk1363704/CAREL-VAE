import json
import random
import re
from collections import defaultdict

from constant import *

mappings = {
  "anger": "5",
  "angry": "5",
  "disgust": "2",
  "fear": "4",
  "happiness": "0",
  "happines": "0",
  "happy": "0",
  "null": "6",
  "sadness": "1",
  "sad": "1",
  "surprise": "3",
  "surprised": "3",
  "excited": "3"
 }

# for reproducible results
random.seed(42)

class MergeDataset(object):
    def __init__(self):
        self.dialogues = dict()
        self.new_dialogues = dict() if not os.path.exists(PROJECT_ABSOLUTE_PATH + '/Chinese_ECPE/new.json') else self.read_json(PROJECT_ABSOLUTE_PATH + '/Chinese_ECPE/new.json')
        self.get_original_dataset()
        self.polish()

    def read_json(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def write_into_file(self, dataset, path):
        with open(path, "w") as write_file:
            json.dump(dataset, write_file, indent=1, sort_keys=True, ensure_ascii=False)

    def print_dialogue(self, dialogue):
        print('class: {}'.format(dialogue['class']))
        print('len: {}'.format(dialogue['len']))
        print('content: ')
        for line in dialogue['content']:
            print(line.strip())

    def get_original_dataset(self):
        folder = PROJECT_ABSOLUTE_PATH + '/Chinese_ECPE'
        for filename in os.listdir(folder):
            if '.txt' in filename and '_num' not in filename:
                with open(folder+'/'+filename) as f:
                    category = filename.replace('.txt', '')
                    lines = f.readlines()
                    index = 0
                    while index < len(lines):
                        tmp_ = lines[index].strip().split(' ')
                        id_, length = tmp_[0], int(tmp_[1])
                        if id_ in self.dialogues:
                            print('Error! Duplicated for {0}'.format(id_))
                        else:
                            dialogue_ = {'class': category, 'len': length, 'content': lines[index + 1: index + 2 + length]}
                            self.dialogues[id_] = dialogue_
                        index = index + 2 + length
        self.write_into_file(self.dialogues, PROJECT_ABSOLUTE_PATH + '/Chinese_ECPE/original.json')

def write_into_file(dataset, path):
    if not os.path.exists(path):
        # Create the directory if it does not exist
        dir_ = os.path.dirname(path)
        os.mkdir(dir_)
    with open(path, "w") as write_file:
        json.dump(dataset, write_file, indent=1, sort_keys=True, ensure_ascii=False)

def write_into_txt(dataset, path):
    if not os.path.exists(path):
        # Create the directory if it does not exist
        dir_ = os.path.dirname(path)
        if not os.path.exists(dir_):
            os.mkdir(dir_)
    with open(path, 'w') as f:
        f.writelines(dataset)

def get_original_dataset(src_path, tar_path):
    dialogues = dict()
    with open(src_path) as f:
        lines = f.readlines()
        index = 0
        while index < len(lines):
            tmp_ = lines[index].strip().split(' ')
            id_, length = tmp_[0], int(tmp_[1])
            if id_ in dialogues:
                print('Error! Duplicated for {0}'.format(id_))
            else:
                dialogue_ = {'len': length, 'content': lines[index + 1: index + 2 + length]}
                dialogues[id_] = dialogue_
            index = index + 2 + length
    write_into_file(dialogues, tar_path)

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

# Find the mapping between nums and tokens, for instance, 1 -> sadness
def mapping(num_path, word_path):
    num_json, word_json = read_json(num_path), read_json(word_path)
    mapping, reverse_mapping = dict(), dict()
    for key, value in num_json.items():
        if key not in word_json:
            print('Error! No dict {} in word_json!'.format(key))
        else:
            num_conversation = value['content'][1:]
            word_conversation = word_json[key]['content'][1:]
            if len(num_conversation) != len(word_conversation):
                print('Error! Length is different for dict {} in num_json!'.format(key))
            else:
                for num_uttr, word_uttr in zip(num_conversation, word_conversation):
                    num_, word_ = num_uttr.split(',')[1], word_uttr.split(',')[1]
                    if num_ not in mapping and word_ not in reverse_mapping:
                        mapping[num_] = word_
                        reverse_mapping[word_] = num_
                    else:
                        if mapping[num_] != word_:
                            print('Conflicted mapping for id: {}, new_word1:{}, old_word2:{}'.format(num_, word_, mapping[num_]))
                        if word_ in reverse_mapping and reverse_mapping[word_] != num_:
                            print('Conflicted reverse_mapping for word: {}, new_id1:{}, old_id2:{}'.format(word_, num_,
                                                                                                     reverse_mapping[word_]))
    dict_ = {'mapping': mapping, 'reverse': reverse_mapping}
    path = PROJECT_ABSOLUTE_PATH + '/domains/THUCTC_multiple/mapping.json'
    write_into_file(dict_, path)

# Convert the tokens into nums.
def convert(src_path, mapping_path, tar_path):
    new_json, reverse_mapping = read_json(src_path), read_json(mapping_path)['reverse']
    for key, value in new_json.items():
        original_sentences = value['content'][1:]
        for i, original_sentence in enumerate(original_sentences):
            tokens = original_sentence.split(',')
            if '&' in tokens[1]:
                tokens[1] = tokens[1].split('&')[1].strip()
            if tokens[1] in reverse_mapping:
                tokens[1] = reverse_mapping[tokens[1]]
            else:
                print('Unexpected id {} in reverse_mapping!'.format(tokens[1]))
            tokens[2] = reverse_mapping[tokens[2]] if tokens[2] in reverse_mapping else tokens[2]
            value['content'][i+1] = ','.join(tokens)
    write_into_file(new_json, tar_path)

# Transform the json into txt files for training and testing.
def transform(src_path):
    train_output_, test_output_ = defaultdict(list), defaultdict(list)
    json_ = read_json(src_path)
    for key, value in json_.items():
        category = value['class']
        train_output_lines, test_output_lines = [], []

        train_output_lines.append(key + ' ' + str(value['len']) + '\n')
        train_output_lines.extend(value['content'])
        train_output_[category].extend(train_output_lines)

        test_output_lines.append(key + ' ' + str(value['len']) + '\n')
        test_output_lines.append(value['content'][0])
        for sentence in value['content'][1:]:
            tokens = sentence.split(',')
            tokens[2] = '-1'
            sentence = ','.join(tokens)
            test_output_lines.append(sentence)
        test_output_[category].extend(test_output_lines)

    for key, value in train_output_.items():
        write_into_txt(value, PROJECT_ABSOLUTE_PATH + '/data/ECPE_new_dataset/{}.txt'.format(key))
    for key, value in test_output_.items():
        write_into_txt(value, PROJECT_ABSOLUTE_PATH + '/data/ECPE_new_dataset/{}_test.txt'.format(key))

def get_RECCON_emotions(file_path, target_path):
    outputs = []
    data_file = open(file_path, encoding="utf8")
    while True:
        line = data_file.readline()
        if not line:
            break
        if re.search("[0-9]{1,4}[\s][0-9]{1,2}", line):
            outputs.append(line)
            doc_len = int(line.strip().split(" ")[1])
            line = data_file.readline()
            outputs.append(line)
            pos_pairs = eval('[' + line.strip() + ']')
            # doc's sentences
            for i in range(doc_len):
                sentence = data_file.readline()
                # sentence_list.append(sentence)
                elements = sentence.strip().split("\t")
                sen_id, sen_emotion, emotion_label, utterance = elements[0], elements[1], elements[2], elements[3]
                utterance = utterance.replace(',', '')
                if sen_emotion not in mappings:
                    print('Find a new emotion {}!'.format(sen_emotion))
                    sen_emotion = "0"
                else:
                    sen_emotion = mappings[sen_emotion]
                emotion_label = mappings[emotion_label] if emotion_label in mappings else emotion_label
                output_sentence = ','.join([sen_id, sen_emotion, emotion_label, utterance]) + '\n'
                outputs.append(output_sentence)
    with open(target_path, 'w') as f:
        f.writelines(outputs)

def get_RECCON_emotions_minusone(file_path, target_path, bow_optimize=False):
    outputs = []
    data_file = open(file_path, encoding="utf8")
    while True:
        line = data_file.readline()
        if not line:
            break
        if re.search("[0-9]{1,4}[\s][0-9]{1,2}", line):
            outputs.append(line)
            doc_len = int(line.strip().split(" ")[1])
            line = data_file.readline()
            outputs.append(line)
            # doc's sentences
            for i in range(doc_len):
                sentence = data_file.readline()
                # sentence_list.append(sentence)
                elements = sentence.strip().split("\t")
                sen_id, sen_emotion, emotion_label, utterance = elements[0], elements[1], elements[2], elements[3]
                if bow_optimize is False:
                    utterance = utterance.replace(',', ' ').replace(" ", "")
                if sen_emotion not in mappings:
                    print('Find a new emotion {}!'.format(sen_emotion))
                    sen_emotion = "0"
                else:
                    sen_emotion = mappings[sen_emotion]
                emotion_label = "-1"
                output_sentence = ','.join([sen_id, sen_emotion, emotion_label, utterance]) + '\n'
                outputs.append(output_sentence)
    with open(target_path, 'w') as f:
        f.writelines(outputs)

def get_bow_en_file(ecpe, reccon, target):
    data_file = open(ecpe, encoding="utf8")
    ecpe_lines = data_file.readlines()
    data_file = open(reccon, encoding="utf8")
    ecpe_lines += data_file.readlines()
    with open(target, 'w') as f:
        f.writelines(ecpe_lines)

def convert_train_to_test(source, target, bow_optimize=False):
    outputs = []
    data_file = open(source, encoding="utf8")
    while True:
        line = data_file.readline()
        if not line:
            break
        if re.search("[0-9]{1,4}[\s][0-9]{1,2}", line):
            outputs.append(line)
            doc_len = int(line.strip().split(" ")[1])
            line = data_file.readline()
            outputs.append(line)
            # doc's sentences
            for i in range(doc_len):
                sentence = data_file.readline()
                elements = sentence.strip().split(",")
                sen_id, sen_emotion, emotion_label, utterance = elements[0], elements[1], elements[2], elements[3]
                if bow_optimize is False:
                    utterance = utterance.replace(',', ' ').replace(" ", "")
                    if sen_emotion not in ['0', '1', '2', '3', '4', '5', '6']:
                        print('Find a new emotion {}!'.format(sen_emotion))
                        sen_emotion = "0"
                else:
                    if sen_emotion not in mappings:
                        print('Find a new emotion {}!'.format(sen_emotion))
                        sen_emotion = "0"
                    else:
                        sen_emotion = mappings[sen_emotion]
                emotion_label = "-1"
                output_sentence = ','.join([sen_id, sen_emotion, emotion_label, utterance]) + '\n'
                outputs.append(output_sentence)
    if bow_optimize is True:
        path = target.replace('.txt', '_optimize.txt')
    else:
        path = target
    with open(path, 'w') as f:
        f.writelines(outputs)

def convert_train_to_num(source):
    outputs = []
    data_file = open(source, encoding="utf8")
    while True:
        line = data_file.readline()
        if not line:
            break
        if re.search("[0-9]{1,4}[\s][0-9]{1,2}", line):
            outputs.append(line)
            doc_len = int(line.strip().split(" ")[1])
            line = data_file.readline()
            outputs.append(line)
            # doc's sentences
            for i in range(doc_len):
                sentence = data_file.readline()
                elements = sentence.strip().split(",")
                sen_id, sen_emotion, emotion_label, utterance = elements[0], elements[1], elements[2], elements[3]
                if sen_emotion not in mappings:
                    print('Find a new emotion {}!'.format(sen_emotion))
                    sen_emotion = "0"
                else:
                    sen_emotion = mappings[sen_emotion]
                emotion_label = mappings[emotion_label] if emotion_label in mappings else emotion_label
                output_sentence = ','.join([sen_id, sen_emotion, emotion_label, utterance]) + '\n'
                outputs.append(output_sentence)
    path = source.replace('.txt', '_num.txt')
    with open(path, 'w') as f:
        f.writelines(outputs)

def combine_the_domains(source_dir, files, target_path):
    output = []
    for file in files:
        with open(source_dir + file) as f:
            output.extend(f.readlines())
    write_into_txt(output, target_path)

def random_pick(source, target, percentage=1.0):
    dicts = []
    data_file = open(source, encoding="utf8")
    while True:
        line = data_file.readline()
        if not line:
            break
        if re.search("[0-9]{1,4}[\s][0-9]{1,2}", line):
            dict_ = []
            dict_.append(line)
            doc_len = int(line.strip().split(" ")[1])
            line = data_file.readline()
            dict_.append(line)
            # doc's sentences
            for i in range(doc_len):
                dict_.append(data_file.readline())
            dicts.append(dict_)
    random.shuffle(dicts)
    random_elements = random.sample(dicts, k=int(percentage * len(dicts)))
    outputs = []
    for ele_ in random_elements:
        outputs.extend(ele_)
    with open(target, 'w') as f:
        f.writelines(outputs)

def filter_positive_order(source):
    outputs = []
    data_file = open(source, encoding="utf8")
    while True:
        line = data_file.readline()
        if not line:
            break
        if re.search("[0-9]{1,4}[\s][0-9]{1,2}", line):
            temp_, entailment_temp = [line], []

            doc_len = int(line.strip().split(" ")[1])
            # line of the entailments
            line = data_file.readline()
            entailments = line.split(', ')

            for entailment in entailments:
                pair = entailment.strip().replace('(', '').replace(')', '').split(',')
                emotion_index, event_index = int(pair[0]), int(pair[1])
                if event_index <= emotion_index:
                    entailment_temp.append(entailment)
                else:
                    print('Found a negative pair!')

            if len(entailment_temp) > 0:
                temp_.append(', '.join(entailment_temp).strip()+'\n')


            # doc's sentences
            for i in range(doc_len):
                sentence = data_file.readline()
                temp_.append(sentence)
            if len(entailment_temp) > 0:
                outputs.extend(temp_)
            else:
                print('None positive is found in this conversation!')

    if 'test' in source:
        path = source.replace('_test.txt', '_order_test.txt')
    else:
        path = source.replace('.txt', '_order.txt')
    with open(path, 'w') as f:
        f.writelines(outputs)


def main():
    # src_path = PROJECT_ABSOLUTE_PATH + '/domains/THUCTC_multiple/society_num.txt'
    # tar_path = PROJECT_ABSOLUTE_PATH + '/domains/THUCTC_multiple/society_num.json'
    # get_original_dataset(src_path, tar_path)
    #
    # src_path = PROJECT_ABSOLUTE_PATH + '/domains/THUCTC_multiple/society.txt'
    # tar_path = PROJECT_ABSOLUTE_PATH + '/domains/THUCTC_multiple/society.json'
    # get_original_dataset(src_path, tar_path)

    # mapping(PROJECT_ABSOLUTE_PATH + '/domains/THUCTC_multiple/society_num.json', PROJECT_ABSOLUTE_PATH + '/domains/THUCTC_multiple/society.json')

    # src_path = PROJECT_ABSOLUTE_PATH + '/data/new.json'
    # tar_path = PROJECT_ABSOLUTE_PATH + '/data/ECPE_new_dataset/new_conversion.json'
    # mapping_path = PROJECT_ABSOLUTE_PATH + '/domains/THUCTC_multiple/mapping.json'
    # convert(src_path, mapping_path, tar_path)

    # src_path = PROJECT_ABSOLUTE_PATH + '/data/ECPE_new_dataset/new_conversion.json'
    # transform(src_path)

    # src_path = PROJECT_ABSOLUTE_PATH + '/domains/RECCON_precessed_data/dailydialog_valid_ECPE_format.txt'
    # target_path = PROJECT_ABSOLUTE_PATH + '/domains/RECCON_precessed_data/dailydialog_valid_num.txt'
    # get_RECCON_emotions(src_path, target_path)

    # src_path = PROJECT_ABSOLUTE_PATH + '/domains/RECCON_precessed_data/dailydialog_test_ECPE_format.txt'
    # target_path = PROJECT_ABSOLUTE_PATH + '/domains/RECCON_precessed_data/dailydialog_test_num_minusone_optimize.txt'
    # get_RECCON_emotions_minusone(src_path, target_path)

    # ecpe_path = PROJECT_ABSOLUTE_PATH + '/data/all_data_pair_en.txt'
    # reccon_path = PROJECT_ABSOLUTE_PATH + '/domains/RECCON_precessed_data/dailydialog_test_num.txt'
    # target_path = PROJECT_ABSOLUTE_PATH + '/data/ecpe_and_reccon_all_data_pair_en.txt'
    # get_bow_en_file(ecpe_path, reccon_path, target_path)

    # source_path = PROJECT_ABSOLUTE_PATH + '/domains/Englishnovel_multiple/war_new.txt'
    # target_path = PROJECT_ABSOLUTE_PATH + '/pair_data/emotion/war_new.txt'
    # convert_train_to_test(source_path, target_path, bow_optimize=True)

    # source_dir = PROJECT_ABSOLUTE_PATH + '/pair_data/emotion/'
    # files = ['adventure_new_optimize.txt', 'biography_optimize.txt', 'history_num_optimize.txt', 'romance_optimize.txt', 'war_new_optimize.txt']
    # target_path = PROJECT_ABSOLUTE_PATH + '/pair_data/emotion/enecpe_optimize.txt'
    # combine_the_domains(source_dir, files, target_path)

    # for source_path in ['adventure_new.txt', 'biography.txt', 'war_new.txt', 'romance.txt']:
    #     source_path = PROJECT_ABSOLUTE_PATH + '/domains/Englishnovel_multiple/{}'.format(source_path)
    #     convert_train_to_num(source_path)

    # source_dir = PROJECT_ABSOLUTE_PATH + '/domains/Englishnovel_multiple/'
    # files = ['history_num.txt', 'adventure_new_num.txt', 'biography_num.txt', 'romance_num.txt',
    #          'war_new_num.txt']
    # target_path = PROJECT_ABSOLUTE_PATH + '/domains/Englishnovel_multiple/enecpe_num.txt'
    # combine_the_domains(source_dir, files, target_path)

    # src_path = PROJECT_ABSOLUTE_PATH + '/domains/RECCON_precessed_data/dailydialog_train_ECPE_format.txt'
    # target_path = PROJECT_ABSOLUTE_PATH + '/domains/RECCON_precessed_data/dailydialog_train_num_minusone_optimize.txt'
    # get_RECCON_emotions_minusone(src_path, target_path, bow_optimize=True)

    # src_path = PROJECT_ABSOLUTE_PATH + '/domains/RECCON_precessed_data/dailydialog_valid_ECPE_format.txt'
    # target_path = PROJECT_ABSOLUTE_PATH + '/domains/RECCON_precessed_data/dailydialog_valid_num_minusone_optimize.txt'
    # get_RECCON_emotions_minusone(src_path, target_path, bow_optimize=True)

    # src_path = PROJECT_ABSOLUTE_PATH + '/domains/RECCON_precessed_data/dailydialog_train_ECPE_format.txt'
    # target_path = PROJECT_ABSOLUTE_PATH + '/domains/RECCON_precessed_data/dailydialog_train_num_minusone.txt'
    # get_RECCON_emotions_minusone(src_path, target_path)
    #
    # src_path = PROJECT_ABSOLUTE_PATH + '/domains/RECCON_precessed_data/dailydialog_valid_ECPE_format.txt'
    # target_path = PROJECT_ABSOLUTE_PATH + '/domains/RECCON_precessed_data/dailydialog_valid_num_minusone.txt'
    # get_RECCON_emotions_minusone(src_path, target_path)

    # source_dir = PROJECT_ABSOLUTE_PATH + '/domains/RECCON_precessed_data/'
    # files = ['dailydialog_test_num.txt', 'dailydialog_train_num.txt', 'dailydialog_valid_num.txt']
    # target_path = PROJECT_ABSOLUTE_PATH + '/domains/Englishnovel_multiple/reccon_num.txt'
    # combine_the_domains(source_dir, files, target_path)

    # source_dir = PROJECT_ABSOLUTE_PATH + '/domains/RECCON_precessed_data/'
    # files = ['dailydialog_test_num_minusone_optimize.txt', 'dailydialog_train_num_minusone_optimize.txt', 'dailydialog_valid_num_minusone_optimize.txt']
    # target_path = PROJECT_ABSOLUTE_PATH + '/pair_data/emotion/reccon_optimize.txt'
    # combine_the_domains(source_dir, files, target_path)

    # source_dir = PROJECT_ABSOLUTE_PATH + '/domains/RECCON_precessed_data/'
    # files = ['dailydialog_test_num_minusone.txt', 'dailydialog_train_num_minusone.txt',
    #          'dailydialog_valid_num_minusone.txt']
    # target_path = PROJECT_ABSOLUTE_PATH + '/pair_data/emotion/reccon.txt'
    # combine_the_domains(source_dir, files, target_path)

    # source_dir = PROJECT_ABSOLUTE_PATH + '/pair_data/emotion/'
    # files = ['adventure_new.txt', 'biography.txt', 'history_num.txt', 'romance.txt', 'war_new.txt']
    # target_path = PROJECT_ABSOLUTE_PATH + '/pair_data/emotion/enecpe.txt'
    # combine_the_domains(source_dir, files, target_path)

    # source_path = PROJECT_ABSOLUTE_PATH + '/pair_data/emotion/reccon.txt'
    # target_path = PROJECT_ABSOLUTE_PATH + '/pair_data/emotion/enecpe_random.txt'
    # random_pick(source_path, target_path, percentage=0.5)
    #
    # source_path = PROJECT_ABSOLUTE_PATH + '/pair_data/emotion/enecpe_optimize.txt'
    # target_path = PROJECT_ABSOLUTE_PATH + '/pair_data/emotion/enecpe_random_optimize.txt'
    # random_pick(source_path, target_path, percentage=0.5)

    paths = ['home.txt', 'home_test.txt', 'education.txt', 'education_test.txt', 'entertainment.txt', 'entertainment_test.txt', 'finance.txt', 'finance_test.txt', 'society.txt', 'society_test.txt']

    for path in paths:
        source_path = PROJECT_ABSOLUTE_PATH + '/data/ECPE_new_dataset/' + path
        filter_positive_order(source_path)


if __name__ == '__main__':
    main()
