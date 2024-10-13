"""
Written by Yujin Huang(Jinx)
Started 6/02/2022 1:26 pm
Last Editted 

Description of the purpose of the code
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from yellowbrick.text import  TSNEVisualizer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sentence_transformers import SentenceTransformer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from sklearn.manifold import TSNE
import plotly.express as px
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

domains = ["Books", "Dvd", "Electronics", "Kitchen"]
book = []
dvd = []
elec = []
kitc = []
for d in domains:
    data_file_train = open("data/amazon/" + d + "/" + d + "train.txt", encoding="utf8")
    data_file_test = open("data/amazon/" + d + "/" + d + "test.txt", encoding="utf8")

    train_data = data_file_train.readlines()
    test_data = data_file_test.readlines()

    if d == "Books":
        book = train_data + test_data
    elif d == "Dvd":
        dvd = train_data + test_data
    elif d == "Electronics":
        elec = train_data + test_data
    elif d == "Kitchen":
        kitc = train_data + test_data

data = book + dvd + elec + kitc

df = pd.DataFrame(data, columns=['doc'])
label = ["Book"] * 2000 + ["Dvd"] * 2000 + ["Electronics"] * 2000 + ["Kitchen"] * 2000
df['label'] = label


embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', cache_folder="model")
X_embedding = embedder.encode(df['doc'].tolist())

# tsne = TSNE(n_components=2, verbose=1,perplexity=30.0, random_state=42)
# z = tsne.fit_transform(X_embedding)
# df["comp-1"] = z[:,0]
# df["comp-2"] = z[:,1]
# sns.scatterplot(x="comp-1", y="comp-2", hue=df['label'].tolist(),
#                 palette=sns.color_palette("hls", 4),
#                 data=df).set(title="T-SNE projection")
# plt.savefig("ama_domain.png")

X_LDA_cau = LDA(n_components=2).fit_transform(X_embedding, y=df['label'].tolist())
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
        x=0.84,  # value must be between 0 to 1.
        y=1,  # value must be between 0 to 1.
        bgcolor = 'rgba(0,0,0,0)'
    ),
    font=dict(
        size=9
    )
)
fig.write_image("ama_domain.png")