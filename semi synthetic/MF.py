import numpy as np
import torch
torch.manual_seed(2020)
from torch import nn
import torch.nn.functional as F
from math import sqrt
import pdb


class MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x_user, x_item, is_training=False):
        user_idx = torch.LongTensor(x_user)
        item_idx = torch.LongTensor(x_item)
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def predict(self, x_user, x_item):
        pred = self.forward(x_user, x_item)
        return pred.detach().numpy()        
        
    def fit(self, x_user, x_item, y, x_test_user, x_test_item, y_test,
        num_epoch=1000, batch_size=512, lr=0.05, lamb=0, gamma = 0, verbose = True):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x_user)
        total_batch = num_sample // batch_size
        best_mse = 1e9
        early_stop = 0  
        early_stop1 = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            if early_stop1 >= 8:
                break
            for idx in range(total_batch):
                # mini-batch training
                    selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                    sub_x_user = x_user[selected_idx]
                    sub_x_item = x_item[selected_idx]
                    sub_y = y[selected_idx]
                    sub_y = torch.Tensor(sub_y)

                    pred, u_emb, v_emb = self.forward(sub_x_user, sub_x_item, True)

                    xent_loss = ((pred-sub_y)**2 + gamma *((u_emb)**2 + (v_emb)**2).sum()).sum()

                    optimizer.zero_grad()
                    xent_loss.backward()
                    optimizer.step()
                    
                    mse = ((self.predict(x_test_user, x_test_item)-y_test)**2).sum()/len(x_test_user)
                    if mse < best_mse:
                        best_mse = mse
                        early_stop = 0
                    else:
                        early_stop += 1
                    if early_stop >= 8:
                        early_stop1 = early_stop
                        break

        print("gamma:{}, mse:{}".format(gamma, mse))