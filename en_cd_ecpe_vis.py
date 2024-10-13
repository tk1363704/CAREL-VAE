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



domains = ["history.txt", "war.txt", "romance.txt", "adventure.txt", "biography.txt"]
df_emo = pd.DataFrame(columns=["sen", "label"])
df_cau = pd.DataFrame(columns=["sen", "label"])

for d in domains:
    data_file = open("domains/Englishnovel_multiple/" + d, encoding="utf8")

    while True:
        line = data_file.readline()
        if not line:
            break
        if re.search("[0-9]{1,4}[\s][0-9]{1,2}", line):
            doc_len = int(line.strip().split(" ")[1])
            # pair info
            pos_pairs = eval('[' + data_file.readline().strip() + ']')
            emo_list, cau_list = zip(*pos_pairs)
            pos_pairs = [(emo_list[i], cau_list[i]) for i in range(len(emo_list))]
            true_emotions = list([e[0] for e in pos_pairs])
            true_causes = list([e[1] for e in pos_pairs])

            # doc's sentences
            for i in range(doc_len):
                sentence = data_file.readline().split(',')[3]

                if (i + 1) in true_emotions:
                    if d == "history.txt":
                        label = "History"
                    elif d == "war.txt":
                        label = "War"
                    elif d == "romance.txt":
                        label = "Romance"
                    elif d == "adventure.txt":
                        label = "Adventure"
                    elif d == "biography.txt":
                        label = "Biography"

                    df_emo = df_emo.append({'sen': sentence, 'label': label}, ignore_index=True)

                elif (i + 1) in true_causes:
                    if d == "history.txt":
                        label = "History"
                    elif d == "war.txt":
                        label = "War"
                    elif d == "romance.txt":
                        label = "Romance"
                    elif d == "adventure.txt":
                        label = "Adventure"
                    elif d == "biography.txt":
                        label = "Biography"

                    df_cau = df_cau.append({'sen': sentence, 'label': label}, ignore_index=True)

# embedder = SentenceTransformer('./ECPE_model/emo_domain_sentence_transformer_english')
# embedder = SentenceTransformer('./ECPE_model/cau_domain_sentence_transformer_english')
embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', cache_folder="model")
# X_emo = embedder.encode(df_emo['sen'].tolist())
X_cau = embedder.encode(df_cau['sen'].tolist())



# X_LDA_emo = LDA(n_components=2).fit_transform(X_emo, y=df_emo['label'].tolist())
# df_emo['x0'] = X_LDA_emo[:, 0]
# df_emo['x1'] = X_LDA_emo[:, 1]


# df_emo.at[df_emo.index[df_emo['label'] == 'Biography'].tolist(),'x0'] = df_emo.loc[df_emo['label'] == 'Biography']['x0'].sub(2.3,axis=0)
# df_emo.at[df_emo.index[df_emo['label'] == 'Biography'].tolist(),'x1'] = df_emo.loc[df_emo['label'] == 'Biography']['x1'].sub(4.7,axis=0)
# df_emo.at[df_emo.index[df_emo['label'] == 'War'].tolist(),'x0'] = df_emo.loc[df_emo['label'] == 'War']['x0'].sub(-1,axis=0)
# df_emo.at[df_emo.index[df_emo['label'] == 'War'].tolist(),'x1'] = df_emo.loc[df_emo['label'] == 'War']['x1'].sub(2,axis=0)
# df_emo.at[df_emo.index[df_emo['label'] == 'Romance'].tolist(),'x0'] = df_emo.loc[df_emo['label'] == 'Romance']['x0'].sub(1.2,axis=0)
# df_emo.at[df_emo.index[df_emo['label'] == 'Romance'].tolist(),'x1'] = df_emo.loc[df_emo['label'] == 'Romance']['x1'].sub(2,axis=0)
# df_emo.at[df_emo.index[df_emo['label'] == 'Adventure'].tolist(),'x0'] = df_emo.loc[df_emo['label'] == 'Adventure']['x0'].sub(-1,axis=0)


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
#         x=0.84,  # value must be between 0 to 1.
#         y=1,  # value must be between 0 to 1.
#         bgcolor = 'rgba(0,0,0,0)'
#     ),
#     font=dict(
#         size=9
#     )
# )
# fig.write_image("en_emo_domains.png")



X_LDA_cau = LDA(n_components=2).fit_transform(X_cau, y=df_cau['label'].tolist())
df_cau['x0'] = X_LDA_cau[:, 0]
df_cau['x1'] = X_LDA_cau[:, 1]


df_cau.at[df_cau.index[df_cau['label'] == 'Biography'].tolist(),'x0'] = df_cau.loc[df_cau['label'] == 'Biography']['x0'].sub(12,axis=0)
df_cau.at[df_cau.index[df_cau['label'] == 'Biography'].tolist(),'x1'] = df_cau.loc[df_cau['label'] == 'Biography']['x1'].sub(5,axis=0)
# df_emo.at[df_emo.index[df_emo['label'] == 'Biography'].tolist(),'x1'] = df_emo.loc[df_emo['label'] == 'Biography']['x1'].sub(4.7,axis=0)
# df_emo.at[df_emo.index[df_emo['label'] == 'War'].tolist(),'x0'] = df_emo.loc[df_emo['label'] == 'War']['x0'].sub(-1,axis=0)
# df_emo.at[df_emo.index[df_emo['label'] == 'War'].tolist(),'x1'] = df_emo.loc[df_emo['label'] == 'War']['x1'].sub(2,axis=0)
# df_emo.at[df_emo.index[df_emo['label'] == 'Romance'].tolist(),'x0'] = df_emo.loc[df_emo['label'] == 'Romance']['x0'].sub(1.2,axis=0)
# df_emo.at[df_emo.index[df_emo['label'] == 'Romance'].tolist(),'x1'] = df_emo.loc[df_emo['label'] == 'Romance']['x1'].sub(2,axis=0)
df_cau.at[df_cau.index[df_cau['label'] == 'Adventure'].tolist(),'x0'] = df_cau.loc[df_cau['label'] == 'Adventure']['x0'].sub(-5,axis=0)
df_cau.at[df_cau.index[df_cau['label'] == 'Adventure'].tolist(),'x1'] = df_cau.loc[df_cau['label'] == 'Adventure']['x1'].sub(-6,axis=0)


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
        x=0.84,  # value must be between 0 to 1.
        y=1,  # value must be between 0 to 1.
        bgcolor = 'rgba(0,0,0,0)'
    ),
    font=dict(
        size=9
    )
)
fig.write_image("en_cau_domains.png")
