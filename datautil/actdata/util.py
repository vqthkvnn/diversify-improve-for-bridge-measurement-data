# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pickle

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from torchvision import transforms
import numpy as np
import seaborn as sns

def act_train():
    return transforms.Compose([
        transforms.ToTensor()
    ])


def loaddata_from_numpy(dataset='dsads', task='cross_people', root_dir='./data/act/'):
    num_class = {
        "0": 32,
        "1": 26,
        "2": 29,
        "3": 32,
        "4": 36,
        "5": 27,
        "6": 32,
        "7": 28,
        "8": 40,
        "9": 30,
        '10': 33,
        "11": 31,
        "12": 31,
        "13": 34,
        "14": 32,
        "15": 35
    }
    if dataset == "target":
        with open('valid_Z24.p', 'rb') as f:
            x = pickle.load(f)
            y_train = pickle.load(f)
            x_train = []
            y_label = []
            for id_, value in enumerate(x):
                for index in range(x.shape[1] // 200):
                    x_train.append(value[index * 200:(index + 1) * 200])
                    y_label.append(float(y_train[id_]))
            x_train = np.array(x_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[2], x_train.shape[1]))
        cy, py, sy = np.array(y_label), np.array(y_label), np.zeros(len(y_label), dtype=float)
    elif dataset == "cm":
        data_x = []
        data_y = []
        with open('train_Z24.p', 'rb') as f:
            xtrain = pickle.load(f)
            ytrain = pickle.load(f)
        with open('valid_Z24.p', 'rb') as f:
            xval = pickle.load(f)
            yval = pickle.load(f)
        train = np.concatenate((xtrain, xval))
        labels = np.concatenate((ytrain, yval))
        x_train, _, y_train, _ = train_test_split(train, labels, random_state=32, test_size=0.5)
        for id_, value in enumerate(x_train):
            for index in range(x_train.shape[1] // 200):
                if data_y.count(y_train[id_]) > num_class[str(y_train[id_])]:
                    continue
                else:
                    data_x.append(value[index * 200:(index + 1) * 200])
                    data_y.append(y_train[id_])
        cy, py, sy = np.array(data_y), np.array(data_y), np.zeros(len(data_y), dtype=float)
        x_train = np.array(data_x)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[2], x_train.shape[1]))
        # raw_data = np.reshape(x_train, (x_train.shape[0], x_train.shape[2] * x_train.shape[1]))
        # tsne_results = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=16).fit_transform(raw_data)
        # df = pd.DataFrame()
        # df["y"] = cy
        # df["comp-1"] = tsne_results[:, 0]
        # df["comp-2"] = tsne_results[:, 1]
        #
        # sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
        #                 palette=sns.color_palette("hls", 16),
        #                 data=df).set(title="Z24 data T-SNE Visual Raw")
        # plt.show()
    else:
        with open('train_Z24.p', 'rb') as f:
            data_x = []
            data_y = []
            x = pickle.load(f)
            y_train = pickle.load(f)
            x_train = []
            y_label = []
            for id_, value in enumerate(x):
                for index in range(x.shape[1] // 200):
                    x_train.append(value[index * 200:(index + 1) * 200])
                    y_label.append(float(y_train[id_]))
                    if data_y.count(y_train[id_]) > num_class[str(y_train[id_])]:
                        continue
                    else:
                        data_x.append(value[index * 200:(index + 1) * 200])
                        data_y.append(y_train[id_])
            x_train = np.array(x_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[2] * x_train.shape[1]))
        cy, py, sy = np.array(y_label), np.array(y_label), np.zeros(len(y_label), dtype=float)
        data_x = np.array(data_x)
        data_x = np.reshape(data_x, (data_x.shape[0], data_x.shape[2] * data_x.shape[1]))
        data_y = np.array(data_y)
        tsne_results = TSNE(n_components=2, learning_rate='auto',init = 'random', perplexity = 16).fit_transform(data_x)
        tsne_results = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
        plt.figure()
        plt.scatter(tsne_results['tsne1'], tsne_results['tsne2'], c=data_y)
        plt.show()
    # if dataset == 'pamap' and task == 'cross_people':
    #     x = np.load(root_dir+dataset+'/'+dataset+'_x1.npy')
    #     ty = np.load(root_dir+dataset+'/'+dataset+'_y1.npy')
    # else:
    #     x = np.load(root_dir+dataset+'/'+dataset+'_x.npy')
    #     ty = np.load(root_dir+dataset+'/'+dataset+'_y.npy')
    # cy, py, sy = ty[:, 0], ty[:, 1], ty[:, 2]
    return x_train, cy, py, sy
