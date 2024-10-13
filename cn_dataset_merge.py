import json

from constant import *
from InquirerPy import inquirer
from InquirerPy.base.control import Choice


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


def main():
    mergeDataset = MergeDataset()


if __name__ == '__main__':
    main()
