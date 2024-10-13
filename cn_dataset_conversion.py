import json
from collections import defaultdict

from constant import *


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

    def polish(self):
        for key, value in self.dialogues.items():
            if key in self.new_dialogues:
                print('{} is existed!'.format(key))
                continue
            # categories = ['society']
            if value['class'] == 'society':
                print('id: {}'.format(key))
                self.print_dialogue(value)
                print('---------------')
                action = inquirer.select(
                    message="Select the correct class that the current piece of news belongs to:",
                    choices=[
                        "society",
                        "home",
                        "finance",
                        "education",
                        "entertainment",
                        "lottery",
                        "sports",
                        "politics",
                        "realEstate",
                        "technology",
                        "other",
                        Choice(value=None, name="Exit"),
                    ],
                    default="society",
                ).execute()
                print('---------------')
                if action is not None:
                    value['class'] = action
                    self.new_dialogues[key] = value
                else:
                    break
        self.write_into_file(self.new_dialogues, PROJECT_ABSOLUTE_PATH + '/Chinese_ECPE/new.json')

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


def main():
    # src_path = PROJECT_ABSOLUTE_PATH + '/domains/THUCTC_multiple/society_num.txt'
    # tar_path = PROJECT_ABSOLUTE_PATH + '/domains/THUCTC_multiple/society_num.json'
    # get_original_dataset(src_path, tar_path)
    #
    # src_path = PROJECT_ABSOLUTE_PATH + '/domains/THUCTC_multiple/society.txt'
    # tar_path = PROJECT_ABSOLUTE_PATH + '/domains/THUCTC_multiple/society.json'
    # get_original_dataset(src_path, tar_path)

    mapping(PROJECT_ABSOLUTE_PATH + '/domains/THUCTC_multiple/society_num.json', PROJECT_ABSOLUTE_PATH + '/domains/THUCTC_multiple/society.json')

    # src_path = PROJECT_ABSOLUTE_PATH + '/data/new.json'
    # tar_path = PROJECT_ABSOLUTE_PATH + '/data/ECPE_new_dataset/new_conversion.json'
    # mapping_path = PROJECT_ABSOLUTE_PATH + '/domains/THUCTC_multiple/mapping.json'
    # convert(src_path, mapping_path, tar_path)

    src_path = PROJECT_ABSOLUTE_PATH + '/data/ECPE_new_dataset/new_conversion.json'
    transform(src_path)

if __name__ == '__main__':
    main()
