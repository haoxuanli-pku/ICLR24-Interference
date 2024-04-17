# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import scipy.sparse as sps
import numpy as np
import torch
torch.manual_seed(2020)
from torch import nn
import torch.nn.functional as F
from math import sqrt
import pdb
import time

from utils import ndcg_func,  recall_func, precision_func
acc_func = lambda x,y: np.sum(x == y) / len(x)
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from collections import defaultdict

mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)

def generate_total_sample(num_user, num_item):
    sample = []
    for i in range(num_user):
        sample.extend([[i,j] for j in range(num_item)])
    return np.array(sample)

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


class MF(nn.Module):
    def __init__(self, num_users, num_items, batch_size, embedding_k=4, *args, **kwargs):
        super(MF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.batch_size = batch_size
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = self.sigmoid(torch.sum(U_emb.mul(V_emb), 1))

        if is_training:
            return out, U_emb, V_emb
        else:
            return out
           
    def fit(self, x, y, 
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                # print(selected_idx)
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb = self.forward(sub_x, True)
                xent_loss = self.xent_func(pred,sub_y)

                #loss = xent_loss

                optimizer.zero_grad()
                xent_loss.backward()
                optimizer.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu().numpy()

class MF_BaseModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF_BaseModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).cuda()
        item_idx = torch.LongTensor(x[:, 1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = self.sigmoid(torch.sum(U_emb.mul(V_emb), 1))

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu()

class NCF_BaseModel(nn.Module):
    """The neural collaborative filtering method.
    """
    
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(NCF_BaseModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, 1, bias = True)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.xent_func = torch.nn.BCELoss()


    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)

        out = self.sigmoid(self.linear_1(z_emb))
        if is_training:
            return torch.squeeze(out), U_emb, V_emb
        else:
            return torch.squeeze(out)        
        
    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu()


class Embedding_Sharing(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(Embedding_Sharing, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.xent_func = torch.nn.BCELoss()


    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)

        if is_training:
            return torch.squeeze(z_emb), U_emb, V_emb
        else:
            return torch.squeeze(z_emb)        
    
    
    
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.linear_1 = torch.nn.Linear(input_size, input_size / 2, bias = False)
        self.linear_2 = torch.nn.Linear(input_size / 2, 1, bias = True)
        self.xent_func = torch.nn.BCELoss()        
    
    def forward(self, x):
        
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.sigmoid(x)
        
        return torch.squeeze(x)             
    
def one_hot(x):
    out = torch.cat([torch.unsqueeze(1-x,1),torch.unsqueeze(x,1)],axis=1)
    return out

def sharpen(x, T):
    temp = x**(1/T)
    return temp / temp.sum(1, keepdim=True)

def calculate_g (x_train, num_user = 290, num_item = 300):
    interference_matrix = np.zeros((num_user, num_item))
    x = np.zeros((num_user, num_item))
    for i in x_train:
        x[i[0],i[1]] = 1
    for i in range(num_user):
        interference_matrix[i,:] += np.sum(x[i,:])# + np.sum(x[:,j])
    for j in range(num_item):
        interference_matrix[:,j] += np.sum(x[:,j])
    
    # interference_matrix -= x
    
    interference_matrix -= (2*x)

    return interference_matrix

class PS(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, low = 0.05, up = 0.95, c = 1, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)    
        self.linear = nn.Linear(2 * embedding_k + 1, 1)
        self.low = low
        self.up = up
        self.c = c
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()
        
    def forward(self, x, g):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        
        g = g.unsqueeze(dim = 1)
        z_emb = torch.cat([U_emb, V_emb, g], axis=1)
        out = torch.clip(self.sigmoid(self.linear(z_emb).squeeze()), self.low, self.up)
        
        return out

    def fit(self, x, thr = 0.8, num_epoch=1000, batch_size=128, lr=0.05, lamb=1e-4,
        tol=1e-4, verbose=True):
        
        obs = sps.csr_matrix((np.ones(len(x)), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray()
                
        interference_matrix = calculate_g(x, self.num_users, self.num_items)
        g_value = interference_matrix[obs == 1]
        g_value_unique = np.unique(g_value[g_value < np.quantile(g_value, thr)])
        g_value_random = np.random.choice(g_value_unique, len(x), replace = True)
        
        g_value = torch.Tensor(g_value).cuda()
        g_value_random = torch.Tensor(g_value_random).cuda()
            
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9
        
        num_sample = len(x)
        total_batch = num_sample // batch_size
        
        early_stop = 0

        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                
                sub_x = x[selected_idx]
                sub_y1 = torch.ones(sub_x.shape[0]).cuda()
                sub_y2 = torch.zeros(sub_x.shape[0]).cuda()
                sub_g1 = g_value[selected_idx]
                sub_g2 = g_value_random[selected_idx]
                
                pred1 = self.forward(sub_x, sub_g1)
                pred2 = self.forward(sub_x, sub_g2)
                
                loss1 = F.binary_cross_entropy(pred1, sub_y1)
                loss2 = F.binary_cross_entropy(pred2, sub_y2)
                
                loss = loss1 + loss2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.detach().cpu().numpy()  
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-Interference-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss
        
        return g_value
    def predict(self, x, g):
        g = g * torch.ones(x.shape[0]).cuda()
        pred = self.c * self.forward(x, g)/ (1 - self.forward(x, g))
        return pred.detach()
        
class MF_N_IPS(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, low = 0.05, up = 0.95, c = 1, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        
        self.propensity_model = PS(num_users, num_items, embedding_k, low = low, up = up, c = c)
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return self.sigmoid(out), U_emb, V_emb
        else:
            return self.sigmoid(out)

    def fit(self, x, y, y_ips = None, g_value = 0, g = 10, thr = 0.8, h = 1, gamma = 0.05,
        num_epoch=1000, batch_size=128, lr=0.05, lamb1=1e-4, lamb2 = 1e-4,
        tol=1e-4, verbose=True):
        
        if sum(g_value) == 0:
            obs = sps.csr_matrix((np.ones(len(x)), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray()

            interference_matrix = calculate_g(x, self.num_users, self.num_items)
            g_value = torch.Tensor(interference_matrix[obs == 1])

        g_value = g_value.cuda()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb2)
        last_loss = 1e9
               
        num_sample = len(x)
        total_batch = num_sample // batch_size
        
        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)        
        
        inv_prop_all = one_over_zl * self.propensity_model.predict(x, g = g)
        print(inv_prop_all)
        early_stop = 0
        y = torch.Tensor(y).cuda()
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_g = g_value[selected_idx]
                # propensity score
                inv_prop = inv_prop_all[selected_idx]              
                
                pred = self.forward(sub_x)                
                
                loss = F.binary_cross_entropy(pred, sub_y,
                    weight = inv_prop, reduction = 'none')

                loss = loss * torch.exp(-(sub_g - g)**2 / (2 * (h**2)))

                loss = torch.mean(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.detach().cpu().numpy() 

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-N-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-N-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-N-IPS] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu().numpy()
    
    def _compute_IPS(self,x,y,y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1

            propensity = np.zeros(len(y))

            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl).cuda()
        return one_over_zl    
        
class MF_N_DR_JL(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, low = 0.05, up = 0.95, c = 1, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.imputation_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        
        self.propensity_model = PS(num_users, num_items, embedding_k, low, up, c = c)        
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, y_ips = None, g_value = 0, G = 1, g = 10, thr = 0.8, h = 1, gamma = 0.05,
        num_epoch=1000, batch_size=128, lr=0.05, lamb1=1e-4, lamb2 = 1e-4,
        tol=1e-4, verbose=True):

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb1)
        optimizer_imputation = torch.optim.Adam(
            self.imputation_model.parameters(), lr=lr, weight_decay=lamb2)

        if sum(g_value) == 0:
            obs = sps.csr_matrix((np.ones(len(x)), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray()

            interference_matrix = calculate_g(x, self.num_users, self.num_items)
            g_value = torch.Tensor(interference_matrix[obs == 1])
            
        g_value = g_value.cuda()
        last_loss = 1e9
               
        num_sample = len(x)
        total_batch = num_sample // batch_size
        
        x_all = generate_total_sample(self.num_users, self.num_items) 
        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)        
        
        inv_prop_all = one_over_zl * self.propensity_model.predict(x, g = g)
        
        early_stop = 0
        y = torch.Tensor(y).cuda()
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_g = g_value[selected_idx]
                # propensity score
                inv_prop = inv_prop_all[selected_idx].cuda()            
                
                pred = self.prediction_model.forward(sub_x)                

                x_sampled = x_all[ul_idxs[G * idx* batch_size: G * (idx+1)*batch_size]] 

                pred_ul = self.prediction_model.forward(x_sampled)

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="none") # o*eui/pui
                
                imputation_y = self.imputation_model.predict(sub_x).cuda()  
                imputation_yul = self.imputation_model.predict(x_sampled).cuda()   
                
                imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="none") # e^ui

                ips_loss = torch.sum((xent_loss - imputation_loss) * torch.exp(-(sub_g - g)**2 / (2 * (h**2))))

                # direct loss
                direct_loss = F.binary_cross_entropy(pred_ul, imputation_yul, reduction="sum") 

                loss = (ips_loss + direct_loss) / x_sampled.shape[0]

                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()
                
                epoch_loss += torch.sum(xent_loss).detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x).cuda()  
                imputation_y = self.imputation_model.forward(sub_x)
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (((e_loss - e_hat_loss) ** 2) * inv_prop)
                imp_loss = torch.sum(imp_loss * torch.exp(-(sub_g - g)**2 / (2 * (h**2))))
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
             
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-N-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-N-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-N-DR-JL] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)
        return pred.detach().cpu().numpy()

    def _compute_IPS(self,x,y,y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1

            propensity = np.zeros(len(y))

            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl).cuda()
        return one_over_zl             


class MF_N_MRDR_JL(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, low = 0.05, up = 0.95, c = 1, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.imputation_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        
        self.propensity_model = PS(num_users, num_items, embedding_k, low, up, c = c)        
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, y_ips = None, g_value = 0, G = 1, g = 10, thr = 0.8, h = 1, gamma = 0.05,
        num_epoch=1000, batch_size=128, lr=0.05, lamb1=1e-4, lamb2 = 1e-4,
        tol=1e-4, verbose=True): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb1)
        optimizer_imputation = torch.optim.Adam(
            self.imputation_model.parameters(), lr=lr, weight_decay=lamb2)

        if sum(g_value) == 0:
            obs = sps.csr_matrix((np.ones(len(x)), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray()
            interference_matrix = calculate_g(x, self.num_users, self.num_items)
            g_value = torch.Tensor(interference_matrix[obs == 1])
        g_value = g_value.cuda()
        last_loss = 1e9
          
        x_all = generate_total_sample(self.num_users, self.num_items) 
        num_sample = len(x)
        total_batch = num_sample // batch_size
        
        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)        
        
        inv_prop_all = one_over_zl# * self.propensity_model.predict(x, g = g)
        
        early_stop = 0
        y = torch.Tensor(y).cuda()
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_g = g_value[selected_idx]
                # propensity score
                inv_prop = inv_prop_all[selected_idx].cuda()          
                
                pred = self.prediction_model.forward(sub_x)                

                x_sampled = x_all[ul_idxs[G * idx* batch_size: G * (idx+1)*batch_size]] 

                pred_ul = self.prediction_model.forward(x_sampled)

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="none") # o*eui/pui
                
                imputation_y = self.imputation_model.predict(sub_x).cuda()  
                imputation_yul = self.imputation_model.predict(x_sampled).cuda()  
                
                imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="none") # e^ui

                ips_loss = torch.sum((xent_loss - imputation_loss) * torch.exp(-(sub_g - g)**2 / (2 * (h**2))))

                # direct loss
                direct_loss = F.binary_cross_entropy(pred_ul, imputation_yul, reduction="sum") 

                loss = (ips_loss + direct_loss) / x_sampled.shape[0]

                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()
                
                epoch_loss += torch.sum(xent_loss).detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x).cuda()  
                imputation_y = self.imputation_model.forward(sub_x)
                
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (((e_loss - e_hat_loss) ** 2) * (inv_prop ** 2 ) * (1 - 1 / inv_prop))
                imp_loss = torch.sum(imp_loss * torch.exp(-(sub_g - g)**2 / (2 * (h**2))))
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
             
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-N-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-N-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-N-DR-JL] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)
        return pred.detach().cpu().numpy()

    def _compute_IPS(self,x,y,y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1

            propensity = np.zeros(len(y))

            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl).cuda()
        return one_over_zl
