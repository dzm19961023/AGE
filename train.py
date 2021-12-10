from __future__ import division
from __future__ import print_function
import os, sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
# For replicating the experiments

# 参数初始化，训练调整

SEED = 42
import argparse
import time
import random
import numpy as np
import scipy.sparse as sp
import torch

# 改变随机数生成器的种子，在调用其他随机数模块函数之前调用此函数
# 设置好种子之后，每次产生的随机数会是同一个
np.random.seed(SEED)
# 设置CPU生成随机数种子，方便下次复现实验结果
torch.manual_seed(SEED)

from torch import optim
import torch.nn.functional as F
from model import LinTrans, LogReg
from optimizer import loss_function
from utils import *
from sklearn.cluster import SpectralClustering, KMeans
from clustering_metric import clustering_metrics
from tqdm import tqdm
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn import metrics
import matplotlib.pyplot as plt

# argparse：一个python模块，命令行选项、参数和子命令解析器
# 可以自动生成帮助和使用手册，并在用户给程序传入无效参数时报出错误信息
parser = argparse.ArgumentParser()
# 添加参数
# class argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True)
parser.add_argument('--gnnlayers', type=int, default=3, help="Number of gnn layers")
parser.add_argument('--linlayers', type=int, default=1, help="Number of hidden layers")
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--dims', type=int, default=[500], help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--upth_st', type=float, default=0.0015, help='Upper Threshold start.')
parser.add_argument('--lowth_st', type=float, default=0.1, help='Lower Threshold start.')
parser.add_argument('--upth_ed', type=float, default=0.001, help='Upper Threshold end.')
parser.add_argument('--lowth_ed', type=float, default=0.5, help='Lower Threshold end.')
parser.add_argument('--upd', type=int, default=10, help='Update epoch.')
parser.add_argument('--bs', type=int, default=10000, help='Batchsize.')
parser.add_argument('--dataset', type=str, default='citeseer', help='type of dataset.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
# 通过parser.parse_args()方法解析参数
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda is True:
    print('Using GPU')
    torch.cuda.manual_seed(SEED)
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

#
def clustering(Cluster, feature, true_labels):
    # np.matmul()返回两个数组的矩阵乘积
    # np.transpose()用于改变序列，在指定变参数的情况下默认为矩阵转置
    f_adj = np.matmul(feature, np.transpose(feature))
    # fit_predict(f_adj) 进行谱聚类
    predict_labels = Cluster.fit_predict(f_adj)

    # 构造clustering_metrics实例
    cm = clustering_metrics(true_labels, predict_labels)
    # 用于评估聚类模型 metrics.davies_bouldin_score()得到的分数越低，说明模型越好，最小值为0
    db = -metrics.davies_bouldin_score(f_adj, predict_labels)
    # 获取训练衡量指标结果：
    acc, nmi, adj = cm.evaluationClusterModelFromLabel(tqdm)

    return db, acc, nmi, adj

# 更新相似矩阵，输出正负样本
def update_similarity(z, upper_threshold, lower_treshold, pos_num, neg_num):
    # 求相似矩阵S
    f_adj = np.matmul(z, np.transpose(z))  # Z*Z的转置
    cosine = f_adj
    # reshape 在不更改数据的情况下为数组赋予新值 -1为模糊控制，固定1列，行数自动控制
    cosine = cosine.reshape([-1,])
    # round 更新正负样本数量值（pos_num neg_num）舍入为整数
    pos_num = round(upper_threshold * len(cosine))
    neg_num = round((1-lower_treshold) * len(cosine))
    # 对列向量排序，并取出其中的内容
    pos_inds = np.argpartition(-cosine, pos_num)[:pos_num]
    neg_inds = np.argpartition(cosine, neg_num)[:neg_num]

    return np.array(pos_inds), np.array(neg_inds)

# 更新阈值
def update_threshold(upper_threshold, lower_treshold, up_eta, low_eta):
    upth = upper_threshold + up_eta
    lowth = lower_treshold + low_eta
    return upth, lowth

# args
def gae_for(args):
    print("Using {} dataset".format(args.dataset))
    # sklearn.cluster.SpectralClustering() 一种降维的方法
    # n_clusters: 切图时降到的维数
    # affinity: 相似矩阵的建立方式 precomputed: 自定义相似矩阵
    if args.dataset == 'cora':
        n_clusters = 7
        Cluster = SpectralClustering(n_clusters=n_clusters, affinity = 'precomputed', random_state=0)
    elif args.dataset == 'citeseer':
        n_clusters = 6
        Cluster = SpectralClustering(n_clusters=n_clusters, affinity = 'precomputed', random_state=0)
    elif args.dataset == 'pubmed':
        n_clusters = 3
        Cluster = SpectralClustering(n_clusters=n_clusters, affinity = 'precomputed', random_state=0)
    elif args.dataset == 'wiki':
        n_clusters = 17
        Cluster = SpectralClustering(n_clusters=n_clusters, affinity = 'precomputed', random_state=0)

    # 载入数据
    # 邻接矩阵，属性矩阵，xxx？训练样本范围，训练样本往后500个索引，测试集索引列表
    adj, features, true_labels, idx_train, idx_val, idx_test = load_data(args.dataset)
    # 利用属性矩阵的大小得到节点数量，属性维度（140,1433）
    n_nodes, feat_dim = features.shape
    # 没看懂？？？
    dims = [feat_dim] + args.dims

    # 设定gnn层数，命令中定义Cora层数为8
    # python train.py --dataset cora --gnnlayers 8 --upth_st 0.011 --lowth_st 0.1 --upth_ed 0.001 --lowth_ed 0.5
    layers = args.linlayers
    # Store original adjacency matrix (without diagonal entries) for later
    # np.newaxis 插入新的维度，得到一个二维数组
    # adj.diagonal()对角线元素
    # adj最终变为
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    # adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)

    n = adj.shape[0]

    adj_norm_s = preprocess_graph(adj, args.gnnlayers, norm='sym', renorm=True)
    sm_fea_s = sp.csr_matrix(features).toarray()

    print('Laplacian Smoothing...')
    for a in adj_norm_s:
        sm_fea_s = a.dot(sm_fea_s)
    adj_1st = (adj + sp.eye(n)).toarray()

    db, best_acc, best_nmi, best_adj = clustering(Cluster, sm_fea_s, true_labels)  # 输入为Cluster, feature, true_labels

    best_cl = db
    adj_label = torch.FloatTensor(adj_1st)

    model = LinTrans(layers, dims)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    sm_fea_s = torch.FloatTensor(sm_fea_s)
    adj_label = adj_label.reshape([-1,])

    if args.cuda:
        model.cuda()
        inx = sm_fea_s.cuda()
        adj_label = adj_label.cuda()

    pos_num = len(adj.indices)
    neg_num = n_nodes*n_nodes-pos_num

    up_eta = (args.upth_ed - args.upth_st) / (args.epochs/args.upd)
    low_eta = (args.lowth_ed - args.lowth_st) / (args.epochs/args.upd)

    pos_inds, neg_inds = update_similarity(normalize(sm_fea_s.numpy()), args.upth_st, args.lowth_st, pos_num, neg_num)
    upth, lowth = update_threshold(args.upth_st, args.lowth_st, up_eta, low_eta)

    bs = min(args.bs, len(pos_inds))
    length = len(pos_inds)

    pos_inds_cuda = torch.LongTensor(pos_inds).cuda()
    print('Start Training...')
    for epoch in tqdm(range(args.epochs)):

        st, ed = 0, bs
        batch_num = 0
        model.train()
        length = len(pos_inds)

        while ( ed <= length ):
            sampled_neg = torch.LongTensor(np.random.choice(neg_inds, size=ed-st)).cuda()
            sampled_inds = torch.cat((pos_inds_cuda[st:ed], sampled_neg), 0)
            t = time.time()
            optimizer.zero_grad()
            xind = sampled_inds // n_nodes
            yind = sampled_inds % n_nodes
            x = torch.index_select(inx, 0, xind)
            y = torch.index_select(inx, 0, yind)
            zx = model(x)
            zy = model(y)
            batch_label = torch.cat((torch.ones(ed-st), torch.zeros(ed-st))).cuda()
            batch_pred = model.dcs(zx, zy)
            loss = loss_function(adj_preds=batch_pred, adj_labels=batch_label, n_nodes=ed-st)

            loss.backward()
            cur_loss = loss.item()
            optimizer.step()

            st = ed
            batch_num += 1
            if ed < length and ed + bs >= length:
                ed += length - ed
            else:
                ed += bs


        if (epoch + 1) % args.upd == 0:
            model.eval()
            mu = model(inx)
            hidden_emb = mu.cpu().data.numpy()
            upth, lowth = update_threshold(upth, lowth, up_eta, low_eta)
            pos_inds, neg_inds = update_similarity(hidden_emb, upth, lowth, pos_num, neg_num)
            bs = min(args.bs, len(pos_inds))
            pos_inds_cuda = torch.LongTensor(pos_inds).cuda()

            tqdm.write("Epoch: {}, train_loss_gae={:.5f}, time={:.5f}".format(
                epoch + 1, cur_loss, time.time() - t))

            db, acc, nmi, adjscore = clustering(Cluster, hidden_emb, true_labels)

            if db >= best_cl:
                best_cl = db
                best_acc = acc
                best_nmi = nmi
                best_adj = adjscore


    tqdm.write("Optimization Finished!")
    tqdm.write('best_acc: {}, best_nmi: {}, best_adj: {}'.format(best_acc, best_nmi, best_adj))


if __name__ == '__main__':
    gae_for(args)