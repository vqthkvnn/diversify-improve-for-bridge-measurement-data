# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from network import act_network
import numpy as np

def get_fea(args):
    net = act_network.ActNetwork(args.dataset)
    return net


def accuracy(network, loader, weights, usedpredict='p'):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    class_predict = []
    features = []
    with torch.no_grad():
        for data in loader:
            x = data[0].cuda().float()
            y = data[1].cuda().long()
            if usedpredict == 'p':
                p, feature = network.predict(x)
                for v in feature:
                    features.append(v)
            else:
                p = network.predict1(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset:
                                        weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.cuda()
            if p.size(1) == 1:
                t = p.gt(0).cpu().detach().numpy()
                class_predict = np.concatenate((class_predict, t))
                correct += (p.gt(0).eq(y).float() *
                            batch_weights.view(-1, 1)).sum().item()
            else:
                t = p.argmax(1).cpu().detach().numpy()
                class_predict = np.concatenate((class_predict, t))
                correct += (p.argmax(1).eq(y).float() *
                            batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()
    labels = loader.dataset.labels
    cr = classification_report(labels, class_predict)
    print(cr)
    cm = confusion_matrix(labels, class_predict)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    disp.ax_.set_title("diversify for Z24 dataset")
    # features = np.array(features)
    # tsne_results = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=16).fit_transform(features)
    # df = pd.DataFrame()
    # df["y"] = labels
    # df["comp-1"] = tsne_results[:, 0]
    # df["comp-2"] = tsne_results[:, 1]
    #
    # sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
    #                 palette=sns.color_palette("hls", 16),
    #                 data=df).set(title="Z24 data T-SNE Visual Classification")
    # tsne_results = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
    # plt.figure()
    # plt.scatter(tsne_results['tsne1'], tsne_results['tsne2'], c=labels, palette=sns.color_palette("hls", 3))
    plt.show()
    return correct / total
