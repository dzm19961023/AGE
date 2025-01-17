import pickle as pkl

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
import sklearn.preprocessing as preprocess


## 工具函數定义

def sample_mask(idx, l):
    """Create mask."""
    # 返回一个给定形状和类型的，用0填充的数组
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    if dataset == 'wiki':
        adj, features, label = load_wiki()
        return adj, features, label, 0, 0, 0

    for i in range(len(names)):
        '''
        fix Pickle incompatibility of numpy arrays between Python 2 and 3
        https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
        '''
        # format 格式化函数，格式化指定的值，并将其插入到字符串的占位符内
        # {}{} 不设置指定位置，按默认顺序
        # rb 二进制格式打开一个文件用于只读
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
            # pickle 提供一个简单的持久化功能，对数据进行序列化or反序列化， 将对象以文件形式放在磁盘上
            # 使用_Unpickler实现反序列化
            u = pkl._Unpickler(rf)
            # encoding参数告诉pickle如何阶码字符串实例
            # 读取Numpy array或者Python2存储的datetime、date、time示例时，使用encoding=‘latin1’
            u.encoding = 'latin1'
            # 从已打开的文件中读取打包后的对象，重建其中特定对象的层次结构并返回
            cur_data = u.load()
            objects.append(cur_data)
        # objects.append(
        #     pkl.load(open("data/ind.{}.{}".format(dataset, names[i]), 'rb')))
    # 分别给这些数据赋值
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    # 返回一堆测试集索引（将文件内容按行替换成int型变量）
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset))
    # 重排序 默认按行排序（行内部变为有序）
    test_idx_range = np.sort(test_idx_reorder)

    # citeseer需要特殊处理一下
    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    # vstack 将矩阵按照行进行拼接 构造属性矩阵
    # tx shape为（140,1433）
    # tolil（）将原矩阵转换为稀疏链表，可以提高访问速度
    features = sp.vstack((allx, tx)).tolil()
    # torch.FloatTensor默认生成32位浮点数
    # np.array 用于创建数组
    features = torch.FloatTensor(np.array(features.todense()))
    # 获得图的邻接矩阵(压缩后)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    # 将除训练集以外的标签按行拼接， 大小（2708，7）
    labels = np.vstack((ally, ty))
    # 没看懂
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # tolist()将矩阵转换换为列表
    # 测试集索引列表
    idx_test = test_idx_range.tolist()
    # 返回的是一个可迭代对象，其中类型为对象
    # 用于选出训练样本的大小
    idx_train = range(len(y))
    # 选出从训练样本结束往后500个索引大小
    idx_val = range(len(y), len(y) + 500)

    # 生成一个内容为bool型的数组
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    # 构造跟label大小一样的矩阵（2708，7）
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    # 取前140个（0,139）
    y_train[train_mask, :] = labels[train_mask, :]
    # 取中间500（140,639）
    y_val[val_mask, :] = labels[val_mask, :]
    # 跟测试集相同
    y_test[test_mask, :] = labels[test_mask, :]

    # np.argmax() 获取指定元素中最大值所对饮的索引
    # np.argmax(labels, 1) 返回真实类别信息
    return adj, features, np.argmax(labels, 1), idx_train, idx_val, idx_test

# wiki数据单独加载
def load_wiki():
    f = open('data/graph.txt', 'r')
    adj, xind, yind = [], [], []
    for line in f.readlines():
        line = line.split()

        xind.append(int(line[0]))
        yind.append(int(line[1]))
        adj.append([int(line[0]), int(line[1])])
    f.close()
    ##print(len(adj))

    f = open('data/group.txt', 'r')
    label = []
    for line in f.readlines():
        line = line.split()
        label.append(int(line[1]))
    f.close()

    f = open('data/tfidf.txt', 'r')
    fea_idx = []
    fea = []
    adj = np.array(adj)
    adj = np.vstack((adj, adj[:, [1, 0]]))
    adj = np.unique(adj, axis=0)

    labelset = np.unique(label)
    labeldict = dict(zip(labelset, range(len(labelset))))
    label = np.array([labeldict[x] for x in label])
    adj = sp.csr_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(len(label), len(label)))

    for line in f.readlines():
        line = line.split()
        fea_idx.append([int(line[0]), int(line[1])])
        fea.append(float(line[2]))
    f.close()

    fea_idx = np.array(fea_idx)
    features = sp.csr_matrix((fea, (fea_idx[:, 0], fea_idx[:, 1])), shape=(len(label), 4973)).toarray()
    scaler = preprocess.MinMaxScaler()
    # features = preprocess.normalize(features, norm='l2')
    features = scaler.fit_transform(features)
    features = torch.FloatTensor(features)

    return adj, features, label


# 返回对应索引
def parse_index_file(filename):
    index = []
    # 打开对应文件名内容
    for line in open(filename):
        # line.strip 用于移除字符串头尾指针指定的字符或字符序列
        # 去除首尾空格
        index.append(int(line.strip()))
    return index


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    # assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def decompose(adj, dataset, norm='sym', renorm=True):
    adj = sp.coo_matrix(adj)
    ident = sp.eye(adj.shape[0])
    if renorm:
        adj_ = adj + ident
    else:
        adj_ = adj

    rowsum = np.array(adj_.sum(1))

    if norm == 'sym':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        laplacian = ident - adj_normalized
    evalue, evector = np.linalg.eig(laplacian.toarray())
    np.save(dataset + ".npy", evalue)
    print(max(evalue))
    exit(1)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    n, bins, patches = ax.hist(evalue, 50, facecolor='g')
    plt.xlabel('Eigenvalues')
    plt.ylabel('Frequncy')
    fig.savefig("eig_renorm_" + dataset + ".png")


# 图的预处理
def preprocess_graph(adj, layer, norm='sym', renorm=True):
    # 还原邻接矩阵
    adj = sp.coo_matrix(adj)
    # 创建特殊矩阵：行数与邻接矩阵相同的对角为1的矩阵
    ident = sp.eye(adj.shape[0])
    if renorm:
        # A+I
        adj_ = adj + ident
    else:
        # A
        adj_ = adj
    # 对A+I求每行的和 所以这为什么就是度了呢= =
    rowsum = np.array(adj_.sum(1))

    if norm == 'sym':
        # 对行和求-1/2次幂并变为对角阵， flatten() 返回折叠为一维的数组
        # 得到度矩阵的-1/2次方
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        # 对邻接矩阵进行归一化 D^(-1/2)*A*D(-1/2)
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        # L = I-A_norm 求归一化laplace矩阵
        laplacian = ident - adj_normalized
    elif norm == 'left':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj_).tocoo()
        laplacian = ident - adj_normalized

    # 构造跟层数一样长的list，8层 2/3怎么来的？为什么是2/3？
    reg = [2 / 3] * (layer)

    # 获取滤波矩阵吗？没看懂
    adjs = []
    for i in range(len(reg)):
        # 获得滤波矩阵， I-kL， 所以根本没有算所谓的λmax啊喂！！！
        adjs.append(ident - (reg[i] * laplacian))
    return adjs


def laplacian(adj):
    rowsum = np.array(adj.sum(1))
    degree_mat = sp.diags(rowsum.flatten())
    lap = degree_mat - adj
    return torch.FloatTensor(lap.toarray())


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score
