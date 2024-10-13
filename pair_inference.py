"""
Written by Yujin Huang(Jinx)
Started 23/12/2021 10:59 pm
Last Editted 

Description of the purpose of the code
"""
import torch
import pandas as pd
import re
import random
import os

from transformers import BertTokenizer, BertModel
from sklearn.metrics import precision_score, recall_score, f1_score


class ECPEDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.pairs = df["pair"]
        self.labels = df["label"].values
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = str(self.pairs[index])
        inputs = self.tokenizer.encode_plus(
            pair,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        instance = {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'labels': torch.FloatTensor([self.labels[index]])
        }

        return instance


class PairClassifier(torch.nn.Module):
    def __init__(self, dropout):
        super(PairClassifier, self).__init__()
        self.bert_model = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext",
                                                    cache_dir="model/hfl_chinese-roberta-wwm-ext", return_dict=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(768, 1)

    def forward(self, input_ids, att_mask, token_type_ids):
        output = self.bert_model(
            input_ids,
            attention_mask=att_mask,
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output


def read_ECPE_data(file_path, test=False):
    data_file = open(file_path, encoding="utf8")
    df = pd.DataFrame(columns=["pair", "label"])
    docs_pair_size = []

    while True:
        line = data_file.readline()
        if not line:
            break
        if re.search("[0-9]{1,4}[\s][0-9]{1,2}", line):
            doc_len = int(line.strip().split(" ")[1])
            sentence_list = []
            # positive pair: (emotion, cause)
            pos_pairs = data_file.readline().strip().split(", ")
            pos_pairs = [eval(x) for x in pos_pairs]
            emotions = list(dict.fromkeys([e[0] for e in pos_pairs]))
            causes = [c[1] for c in pos_pairs]

            # negative pair: (emotion, non-cause)
            sen_ids = [i + 1 for i in range(doc_len)]
            sen_ids = [i for i in sen_ids if i not in causes]
            neg_pairs = []
            for e in emotions:
                for non_c in sen_ids:
                    neg_pair = (e, non_c)
                    neg_pairs.append(neg_pair)
            #  training proces
            if not test:
                pos_pairs_size = len(pos_pairs)
                if pos_pairs_size > len(neg_pairs):
                    pos_pairs_size = len(neg_pairs)
                    neg_pairs = random.sample(neg_pairs, pos_pairs_size)
                else:
                    neg_pairs = random.sample(neg_pairs, pos_pairs_size)
            # all_poss_pairs = list(combinations(range(0, doc_len), 2))
            # neg_pairs = random.sample([x for x in all_poss_pairs if x not in list(tuple(sorted(t)) for t in pos_pairs)],
            #                           len(pos_pairs))

            for i in range(doc_len):
                sentence = data_file.readline()
                sentence_list.append(sentence)

            for pos_p in pos_pairs:
                # true emotion-cause pair
                emo_sen_id = pos_p[0]
                cau_sen_id = pos_p[1]
                pos_sen_pair = sentence_list[emo_sen_id - 1].strip().split(",")[3].replace(" ", "") + "[SEP]" + \
                               sentence_list[cau_sen_id - 1].strip().split(",")[3].replace(" ", "")
                df = df.append({'pair': pos_sen_pair, 'label': 1}, ignore_index=True)

            if neg_pairs:
                for neg_p in neg_pairs:
                    # false emotion-cause pair
                    emo_sen_id = neg_p[0]
                    cau_sen_id = neg_p[1]
                    neg_sen_pair = sentence_list[emo_sen_id - 1].strip().split(",")[3].replace(" ", "") + "[SEP]" + \
                                   sentence_list[cau_sen_id - 1].strip().split(",")[3].replace(" ", "")
                    df = df.append({'pair': neg_sen_pair, 'label': 0}, ignore_index=True)

            docs_pair_size.append(len(pos_pairs) + len(neg_pairs))

    return df, docs_pair_size


def load_ckp(checkpoint_path, model):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_path)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # return model, optimizer, epoch value, min validation loss
    return model


def predict(test_df, test_loader, model, path_true, path_pred):
    test_df.to_pickle(path_true)

    model.eval()
    with torch.no_grad():
        # test the model
        print('------------- Test Start -------------------------\n')
        for batch_index, data in enumerate(test_loader):
            ids = data['input_ids'].to(device, dtype=torch.long)
            att_mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.float)

            outputs = model(ids, att_mask, token_type_ids)
            final_outputs = torch.sigmoid(outputs).cpu().detach().numpy().round().tolist()
            labels = labels.cpu().detach().numpy().tolist()

            # precision, recall, and f1 score
            p = precision_score(labels, final_outputs, average="binary")
            r = recall_score(labels, final_outputs, average="binary")
            f1 = f1_score(labels, final_outputs, average="binary")
            print("pair precision: {:.4f}, recall: {:.4f}, f1 socre: {:.4f}\n".format(p, r, f1))

            final_outputs = [l[0] for l in final_outputs]
            test_df['label'] = final_outputs
            test_df.to_pickle(path_pred)


MAX_LEN = 128
DROPOUT = 0.1
best_model_path = "ECPE_model/best_model"
test_path = "domains/THUCTC_multiple/finance.txt"
MODEL_ID = "90217c89-92c0-4624-9731-32f3f954f433"
pair_true_path = "pair_data/ec_pair/" + MODEL_ID + "_true.pkl"
pair_pred_path = "pair_data/ec_pair/" + MODEL_ID + "_pred.pkl"

# load test data
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext', cache_dir="model/hfl_chinese-roberta-wwm-ext")
model = PairClassifier(DROPOUT)
model.to(device)
model = load_ckp(os.path.join(best_model_path, "best_model_" + MODEL_ID + ".pt"), model)

test_df, test_docs_pair_size = read_ECPE_data(test_path, test=True)
test_dataset = ECPEDataset(test_df, tokenizer, MAX_LEN)
test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=test_df.shape[0],
                                               shuffle=False,
                                               num_workers=0
                                               )

predict(test_df, test_data_loader, model, pair_true_path, pair_pred_path)
