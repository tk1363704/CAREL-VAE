"""
Written by Yujin Huang(Jinx)
Started 30/09/2021 4:59 pm
Last Editted 

Description of the purpose of the code
"""
import os.path

import ECPE_dataset
import numpy as np
import warnings
from collections import Counter
from lib.dcc import IDEC
from scipy.stats import chi2_contingency

# setting for loading data and suppressing unnecessary warning
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
np.set_printoptions(suppress=True)
warnings.simplefilter(action='ignore', category=FutureWarning)

# obtain model information and load the cause cluster model
model_name = "Raw_k25_c6000_n5_e100_cause_cluster.pt"
model_path = os.path.join("./ECPE_model/cluster", model_name)
K = int(model_name.split("_")[1][1:])
fine_tune_embedding = True

idec = IDEC(input_dim=768, z_dim=10, n_clusters=K, encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500],
            activation="relu", dropout=0)
idec.load_model(model_path)

# load the test data
X, test_X, y, test_y = ECPE_dataset.load(fine_tune=fine_tune_embedding)

# obtain the predictions
y_pred, train_purity = idec.predict(X, y)
y = y.data.cpu().numpy()

# construct a Contingency Table for each cluster
alpha = 0.05
cluster_ids = set(y_pred)
clu_emo_map = np.zeros((len(cluster_ids), 8)).astype(np.float64)
clu_emo_map[:, 0] = np.array(list(cluster_ids)).T

print("Model name:", model_name)

for c_id in cluster_ids:
    # index for a specific cluster
    row_index = np.where(clu_emo_map[:, 0] == c_id)[0]

    # 0:happiness, 1:sadness, 2:disgust, 3:surprise, 4:fear, 5:anger, 6:none
    emo_labels_c = Counter(l for l, c in zip(y, y_pred) if c == c_id)
    emo_labels_non_c = Counter(l for l, c in zip(y, y_pred) if c != c_id)

    print("\n","\n","cluster id: ",c_id)
    print("emo_labels_c: ",emo_labels_c)
    print("emo_labels_non_c: ",emo_labels_non_c,"\n")

    for e in range(7):
        con_table = [[emo_labels_c[e], (sum(emo_labels_c.values()) - emo_labels_c[e])],
                     [emo_labels_non_c[e], (sum(emo_labels_non_c.values()) - emo_labels_non_c[e])]]

        # Chi-square test of independence of variables
        stat, p, dof, expected = chi2_contingency(con_table)

        if np.any(expected < 5):
            print("not reliable")
            clu_emo_map[row_index, e+1] = -1
            continue

        print("emotion:", e, expected)

        # 1 for dependent and 0 for independent
        if p <= alpha:
            clu_emo_map[row_index, e+1] = stat
            # print('Dependent (reject H0)')
        else:
            clu_emo_map[row_index, e+1] = 0
            # print('Independent (H0 holds true)')

# obtain the predictions of test data
test_y_pred, test_purity = idec.predict(test_X, test_y)

# save results
train_purity.to_csv('./mapping/'+ model_name.split('.')[0] + '_train_purity.csv', index=False)

np.save('./mapping/'+ model_name.split('.')[0] + '_cluster_emotion.npy', clu_emo_map)
np.save('./mapping/'+ model_name.split('.')[0] + '_test_pred.npy', test_y_pred)

