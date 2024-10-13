"""
Written by Yujin Huang(Jinx)
Started 5/02/2022 7:47 pm
Last Editted 

Description of the purpose of the code
"""
import pandas as pd
import re
import jieba
import matplotlib.pyplot as plt
import plotly.express as px
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from yellowbrick.text import UMAPVisualizer, TSNEVisualizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sentence_transformers import SentenceTransformer

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def tokenize_zh(text):
    filter_zh = re.compile(u'[^\u4E00-\u9FA5]')  # non-Chinese unicode range
    text = filter_zh.sub(r'', text)  # remove all non-Chinese characters
    words = jieba.lcut(text)
    return words


domains = ["society.txt", "entertainment.txt", "home.txt", "education.txt", "finance.txt"]
df = pd.DataFrame(columns=["doc", "label"])

for d in domains:
    data_file = open("domains/THUCTC_multiple/" + d, encoding="utf8")

    while True:
        line = data_file.readline()
        if not line:
            break
        if re.search("[0-9]{1,4}[\s][0-9]{1,2}", line):
            doc_len = int(line.strip().split(" ")[1])
            # pair info
            data_file.readline().strip().split(", ")
            content = ''
            # doc's sentences
            for i in range(doc_len):
                sentence = data_file.readline().split(',')[3].replace(" ","")
                content += sentence

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

            df = df.append({'doc': content, 'label': label}, ignore_index=True)

# vectorizer = TfidfVectorizer(tokenizer=tokenize_zh)
# X = vectorizer.fit_transform(df['doc'].tolist())

embedder = SentenceTransformer('./ECPE_model/doc_domain_sentence_transformer_chinese')
doc_embeddings = embedder.encode(df['doc'].tolist())


# pca = PCA(n_components=2, random_state=42)
# pca_vecs = pca.fit_transform(doc_embeddings)
# x0 = pca_vecs[:, 0]
# x1 = pca_vecs[:, 1]
# df['x0'] = x0
# df['x1'] = x1
# plt.xlabel("X0", fontdict={"fontsize": 16})
# plt.ylabel("X1", fontdict={"fontsize": 16})
# # create scatter plot with seaborn, where hue is the class used to group the data
# sns.scatterplot(data=df, x='x0', y='x1', hue='label', palette="viridis")
# plt.show()
# plt.savefig("domain_pca_64_1000.png")




# tsne = TSNEVisualizer()
# tsne.fit(doc_embeddings, df['label'].tolist())
# tsne.show()
# plt.savefig("domain_tsne_64_1000.png")
#
# umap = UMAPVisualizer()
# umap.fit(doc_embeddings, df['label'].tolist())
# umap.show()


# tsne = TSNE(n_components=2, verbose=1,perplexity=30.0, random_state=42)
# z = tsne.fit_transform(doc_embeddings)
# df["comp-1"] = z[:,0]
# df["comp-2"] = z[:,1]
# sns.scatterplot(x="comp-1", y="comp-2", hue=df['label'].tolist(),
#                 palette=sns.color_palette("hls", 5),
#                 data=df).set(title="T-SNE projection")
# plt.savefig("domain_tsne.png")


X_LDA_cau = LDA(n_components=2).fit_transform(doc_embeddings, y=df['label'].tolist())
df['x0'] = X_LDA_cau[:, 0]
df['x1'] = X_LDA_cau[:, 1]
fig = px.scatter(df, x="x0", y="x1", color="label",template="simple_white")
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
fig.write_image("doc_domains.png")

