
import torch
from scipy.io import loadmat
from torch_geometric.data import Data

class Set_division(object):
    def __init__(self, args):
        self.args = args

    def train_test(self):
        args = self.args

        # 保存数据集
        self.datasets_train, self.datasets_test = {}, {}

        # 训练集
        data_train = torch.tensor(loadmat('./mat/' + args.data_save)['x_train'], dtype=torch.float)
        label_train = torch.tensor(loadmat('./mat/' + args.data_save)['y_train'], dtype=torch.long).squeeze(0)
        edge_index_train = torch.tensor(loadmat('./mat/' + args.data_save)['edge_index_train'], dtype=torch.long)
        edge_attr_train = torch.tensor(loadmat('./mat/' + args.data_save)['edge_attr_train'],
                                       dtype=torch.float).squeeze(0)
        self.datasets_train = Data(x=data_train, y=label_train, edge_index=edge_index_train,
                                   edge_attr=edge_attr_train)

        # 测试集
        data_test = torch.tensor(loadmat('./mat/' + args.data_save)['x_test'], dtype=torch.float)
        label_test = torch.tensor(loadmat('./mat/' + args.data_save)['y_test'], dtype=torch.long).squeeze(0)
        edge_index_test = torch.tensor(loadmat('./mat/' + args.data_save)['edge_index_test'], dtype=torch.long)
        edge_attr_test = torch.tensor(loadmat('./mat/' + args.data_save)['edge_attr_test'],
                                      dtype=torch.float).squeeze(0)
        self.datasets_test = Data(x=data_test, y=label_test, edge_index=edge_index_test,
                                  edge_attr=edge_attr_test)

        return self.datasets_train, self.datasets_test