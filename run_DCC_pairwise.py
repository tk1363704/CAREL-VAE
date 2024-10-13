import sys
import os
import torch.utils.data
import numpy as np
import pandas as pd
import argparse
import time
from lib.dcc import IDEC
from lib.datasets import MNIST, FashionMNIST, Reuters
from lib.utils import transitive_closure, generate_random_pair, generate_random_pair_knn
import warnings
import ECPE_dataset
import seaborn
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import chi2_contingency


np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.setrecursionlimit(50000)  # may reach recursion limitation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pairwise MNIST Example')
    parser.add_argument('--lr', type=float, default=0.001, metavar='N',
                        help='learning rate for training (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--update-interval', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--pretrain', type=str, default="../model/mnist_sdae_weights.pt", metavar='N',
                        help='directory for pre-trained weights')
    parser.add_argument('--data', type=str, default="MNIST", metavar='N', help='dataset(MNIST, Fashion, Reuters)')
    parser.add_argument('--without_pretrain', action='store_false')
    parser.add_argument('--without_kmeans', action='store_false')
    parser.add_argument('--noisy', type=float, default=0.0, metavar='N',
                        help='noisy constraints rate for training (default: 0.0)')
    parser.add_argument('--plotting', action='store_true')
    args = parser.parse_args()

    # hyperparameter settings
    args.data = "EC_Pair"
    AE_pretrained = False
    fine_tune = True
    args.epochs=100
    args.lr=0.001
    K=20
    num_constraints = 10000
    num_neighbors = 3
    ml_penalty, cl_penalty = 0.1, 1


    idec = IDEC(input_dim=768, z_dim=10, n_clusters=10,
                encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", dropout=0)

    

    if args.data == "Reuters":
        reuters_train = Reuters('./data/reuters', train=True, download=False)
        reuters_test = Reuters('./data/reuters', train=False)
        X = reuters_train.train_data
        y = reuters_train.train_labels
        test_X = reuters_test.test_data
        test_y = reuters_test.test_labels
        args.pretrain = "../model/reuters10k_sdae_weights.pt"
        idec = IDEC(input_dim=2000, z_dim=10, n_clusters=4,
                    encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", dropout=0)
    elif args.data == "EC_Pair":
        X, test_X, y, test_y = ECPE_dataset.load(fine_tune=fine_tune)
        args.pretrain = "./AE_weight/sdae_ECPE_weight_pretrain.pt"
        idec = IDEC(input_dim=768, z_dim=10, n_clusters=K,
                    encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", dropout=0)

    model_tag = "Raw"
    if AE_pretrained:
        model_tag = "Pretrain"
        idec.load_model(args.pretrain)

    init_tag = "Random"
    if args.without_kmeans:
        init_tag = "KMeans"

    # Print Network Structure
    print(idec)

    # Construct Constraints
    ml_ind1, ml_ind2, cl_ind1, cl_ind2 = generate_random_pair_knn(X, y, num=num_constraints, k=num_neighbors)
    ml_ind1, ml_ind2, cl_ind1, cl_ind2 = transitive_closure(ml_ind1, ml_ind2, cl_ind1, cl_ind2, X.shape[0])

    ml_ind1 = ml_ind1[:num_constraints]
    ml_ind2 = ml_ind2[:num_constraints]
    cl_ind1 = cl_ind1[:num_constraints]
    cl_ind2 = cl_ind2[:num_constraints]

    plotting_dir = ""
    if args.plotting:

        dir_name = args.data + "_" + model_tag + "_" + init_tag + "_%d" % num_constraints
        if args.noisy > 0:
            dir_name += "_Noisy_%d%%" % (int(args.noisy * 100))
        dir_name += "_" + time.strftime("%Y%m%d-%H%M")
        plotting_dir = "./plotting/%s" % dir_name
        if not os.path.exists(plotting_dir):
            os.makedirs(plotting_dir)

        mldf = pd.DataFrame(data=[ml_ind1, ml_ind2]).T
        mldf.to_pickle(os.path.join(plotting_dir, "mustlinks.pkl"))
        cldf = pd.DataFrame(data=[cl_ind1, cl_ind2]).T
        cldf.to_pickle(os.path.join(plotting_dir, "cannotlinks.pkl"))

    if args.noisy > 0:
        nml_ind1, nml_ind2, ncl_ind1, ncl_ind2 = generate_random_pair(y, num_constraints * 2)
        ncl_ind1, ncl_ind2, nml_ind1, nml_ind2 = transitive_closure(nml_ind1, nml_ind2, ncl_ind1, ncl_ind2, X.shape[0])

        nml_ind1 = nml_ind1[:int(ml_ind1.size * args.noisy)]
        nml_ind2 = nml_ind2[:int(ml_ind2.size * args.noisy)]
        ncl_ind1 = ncl_ind1[:int(cl_ind1.size * args.noisy)]
        ncl_ind2 = ncl_ind2[:int(cl_ind2.size * args.noisy)]

        if plotting_dir:
            nmldf = pd.DataFrame(data=[nml_ind1, nml_ind2]).T
            nmldf.to_pickle(os.path.join(plotting_dir, "noisymustlinks.pkl"))
            ncldf = pd.DataFrame(data=[ncl_ind1, ncl_ind2]).T
            ncldf.to_pickle(os.path.join(plotting_dir, "noisycannotlinks.pkl"))

        ml_ind1 = np.append(ml_ind1, nml_ind1)
        ml_ind2 = np.append(ml_ind2, nml_ind2)
        cl_ind1 = np.append(cl_ind1, ncl_ind1)
        cl_ind2 = np.append(cl_ind2, ncl_ind2)

    anchor, positive, negative = np.array([]), np.array([]), np.array([])
    instance_guidance = torch.zeros(X.shape[0]).cuda()
    use_global = False

    # Train Neural Network
    train_acc, train_nmi, train_ari, train_purity, epo = idec.fit(anchor, positive, negative, ml_ind1, ml_ind2, cl_ind1,
                                                                  cl_ind2,
                                                                  instance_guidance, use_global, ml_penalty, cl_penalty,
                                                                  X, y,
                                                                  lr=args.lr, batch_size=args.batch_size,
                                                                  num_epochs=args.epochs,
                                                                  update_interval=args.update_interval, tol=1 * 1e-2,
                                                                  use_kmeans=args.without_kmeans, plotting=plotting_dir)

    # Validation for hyperparameters tuning
    # _, valid_purity = idec.predict(valid_X, valid_y)

    # Make Predictions
    _, test_purity = idec.predict(test_X, test_y)

    # Report Results
    # print("ACC:", train_acc)
    # print("NMI:", train_nmi)
    # print("ARI:", train_ari)
    # print("train purity:", train_purity)
    print("\n","Epochs:", epo)
    # print("testAcc:", test_acc)
    # print("testNMI:", test_nmi)
    # print("testARI:", test_ari)
    # print("test purity:", test_purity)
    print("ML Closure:", ml_ind1.shape[0])
    print("CL Closure:", cl_ind1.shape[0])
    print("K =", K)
    print("# neighbors =", num_neighbors)
    print("Learning rate =", args.lr, "\n")

    print("Train: ","\n",train_purity,"\n")
    print("Test: ","\n",test_purity,"\n")

    fg_train = seaborn.catplot(x='cluster', y='purity', hue='label', kind='bar', data=train_purity, dodge=False)
    plt.axhline(y=0.8, color='r', linestyle='--')
    plt.savefig(model_tag+"_k"+str(K)+"_c"+str(num_constraints)+"_n"+str(num_neighbors)+"_train_result.png")
    plt.title("Train", pad=-100, fontsize=9)
    plt.show()

    # fg_train = seaborn.catplot(x='cluster', y='purity', hue='label', kind='bar', data=valid_purity, dodge=False)
    # plt.axhline(y=0.8, color='r', linestyle='--')
    # plt.savefig(model_tag+"_k"+str(K)+"_c"+str(num_constraints)+"_n"+str(num_neighbors)+"_validation_result.png")
    # plt.title("Validation", pad=-100, fontsize=9)
    # plt.show()

    # fg_test = seaborn.catplot(x='cluster', y='purity', hue='label', kind='bar', data=test_purity, dodge=False)
    # plt.axhline(y=0.8, color='r', linestyle='--')
    # plt.savefig(model_tag+"_k"+str(K)+"_c"+str(num_constraints)+"_n"+str(num_neighbors)+"_test_result.png")
    # plt.title("Test", pad=-100, fontsize=9)
    # plt.show()


    # Performa causal discovery
    y_pred, train_purity = idec.predict(X, y)
    y = y.data.cpu().numpy()

    alpha = 0.05
    cluster_ids = set(y_pred)
    clu_emo_map = np.zeros((len(cluster_ids), 8)).astype(np.float64)
    clu_emo_map[:, 0] = np.array(list(cluster_ids)).T

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

    remove = []
    invalid = np.array([-1, -1, -1, -1, -1, -1, -1])
    for index, row in enumerate(clu_emo_map):
        if np.array_equal(row[1:], invalid):
            remove.append(index)

    filtered_mapping = np.delete(clu_emo_map, remove, 0)
    print("\nValid clusters:", filtered_mapping.shape[0])
    clu_ids = filtered_mapping[:, 0].astype(int)
    clu_emotion = np.argmax(filtered_mapping[:, 1:], axis=1)
    filtered_mapping = np.column_stack((clu_ids.T, clu_emotion.T))
    print("Cluster-emotion mapping:\n", filtered_mapping)

    idec.save_model(
        "./ECPE_model/cluster/" + model_tag + "_k" + str(K) + "_c" + str(num_constraints) + "_n" + str(
            num_neighbors) + "_e" + str(args.epochs) + "_cause_cluster.pt")