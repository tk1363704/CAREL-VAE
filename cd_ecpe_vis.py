"""
Written by Yujin Huang(Jinx)
Started 6/02/2022 6:09 pm
Last Editted 

Description of the purpose of the code
"""
import pandas as pd
import re
import jieba
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sentence_transformers import SentenceTransformer
import plotly.express as px
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def tokenize_zh(text):
    filter_zh = re.compile(u'[^\u4E00-\u9FA5]')  # non-Chinese unicode range
    text = filter_zh.sub(r'', text)  # remove all non-Chinese characters
    words = jieba.lcut(text)
    return words


domains = ["society.txt", "entertainment.txt", "home.txt", "education.txt", "finance.txt"]
df_emo = pd.DataFrame(columns=["sen", "label"])
df_cau = pd.DataFrame(columns=["sen", "label"])

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

                if (i + 1) in true_emotions:
                    if d == "society.txt":
                        label = "Society"
                    elif d == "entertainment.txt":
                        label = "Entertainment"
                    elif d == "home.txt":
                        label = "Home"
                    elif d == "education.txt":
                        label = "Education"
                    elif d == "finance.txt":
                        label = "Finance"

                    df_emo = df_emo.append({'sen': sentence, 'label': label}, ignore_index=True)

                elif (i + 1) in true_causes:
                    if d == "society.txt":
                        label = "Society"
                    elif d == "entertainment.txt":
                        label = "Entertainment"
                    elif d == "home.txt":
                        label = "Home"
                    elif d == "education.txt":
                        label = "Education"
                    elif d == "finance.txt":
                        label = "Finance"

                    df_cau = df_cau.append({'sen': sentence, 'label': label}, ignore_index=True)

embedder = SentenceTransformer('./ECPE_model/cau_domain_sentence_transformer_chinese')
# embedder = SentenceTransformer('cyclone/simcse-chinese-roberta-wwm-ext', cache_folder="model")
# X_emo = embedder.encode(df_emo['sen'].tolist())
X_cau = embedder.encode(df_cau['sen'].tolist())

# vectorizer = TfidfVectorizer(tokenizer=tokenize_zh)
# X = vectorizer.fit_transform(df_cau['sen'].tolist())

# X_LDA_emo = LDA(n_components=2).fit_transform(X_emo, y=df_emo['label'].tolist())
# df_emo['x0'] = X_LDA_emo[:, 0]
# df_emo['x1'] = X_LDA_emo[:, 1]
# fig = px.scatter(df_emo, x="x0", y="x1", color="label",template="simple_white")
# fig.update_xaxes(mirror=True,visible=True,title_standoff =0)
# fig.update_yaxes(mirror=True,visible=True,title_standoff =0)
# fig.update_layout(
#     xaxis_title="tsne-1",
#     yaxis_title="tsne-2",
#     legend_title="Domain",
#     xaxis = dict(
#         tickmode = 'linear',
#         tick0 = -10,
#         dtick = 2
#     ),
#     yaxis = dict(
#         tickmode = 'linear',
#         tick0 = -10,
#         dtick = 2
#     ),
#     legend=dict(
#         x=0.81,  # value must be between 0 to 1.
#         y=1,  # value must be between 0 to 1.
#         bgcolor = 'rgba(0,0,0,0)'
#     ),
#     font=dict(
#         size=9
#     )
# )
# fig.write_image("cemo_domain.png")



X_LDA_cau = LDA(n_components=2).fit_transform(X_cau, y=df_cau['label'].tolist())
df_cau['x0'] = X_LDA_cau[:, 0]
df_cau['x1'] = X_LDA_cau[:, 1]
fig = px.scatter(df_cau, x="x0", y="x1", color="label",template="simple_white")
fig.update_xaxes(mirror=True,visible=True,title_standoff =0)
fig.update_yaxes(mirror=True,visible=True,title_standoff =0)
fig.update_layout(
    xaxis_title="tsne-1",
    yaxis_title="tsne-2",
    legend_title="Domain",
    xaxis = dict(
        tickmode = 'linear',
        tick0 = -10,
        dtick = 2
    ),
    yaxis = dict(
        tickmode = 'linear',
        tick0 = -10,
        dtick = 2
    ),
    legend=dict(
        x=0.81,  # value must be between 0 to 1.
        y=1,  # value must be between 0 to 1.
        bgcolor = 'rgba(0,0,0,0)'
    ),
    font=dict(
        size=9
    )
)
fig.write_image("cau_domains.png")
