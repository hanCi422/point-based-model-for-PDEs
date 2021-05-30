"""
Modified from PointConv: https://github.com/DylanWusee/pointconv
Author: Ning Hua
Date: May 2021
"""

import numpy as np

import scipy.io

class Darcyflow_2d():
    def __init__(self, root, r, nsample):
        self.dataset = scipy.io.loadmat(root)
        h = int(((421-1)/r)+1)
        s = h
        self.x = self.dataset['coeff'][:nsample, ::r, ::r][:, :s, :s][..., None]
        # self.x_smooth = self.dataset['Kcoeff'][:nsample, ::r, ::r][:, :s, :s][..., None]
        # self.x_gradx = self.dataset['Kcoeff_x'][:nsample, ::r, ::r][:, :s, :s][..., None]
        # self.x_grady = self.dataset['Kcoeff_y'][:nsample, ::r, ::r][:, :s, :s][..., None]
        # self.x = np.concatenate((self.x, self.x_smooth, self.x_gradx, self.x_grady), -1)
        self.y = self.dataset['sol'][:nsample, ::r, ::r][:, :s, :s][..., None]
        grids = []
        grids.append(np.linspace(0, 1, s))
        grids.append(np.linspace(0, 1, s))
        grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
        grid = grid.reshape(1, s, s, 2)
        grid_add3 = np.zeros_like(grid)
        grid = np.concatenate((grid, grid_add3), -1)[..., :-1]
        self.x = np.concatenate((grid.repeat(nsample, 0), self.x), -1)
        # self.x = self.x.reshape(self.x.shape[0], -1, 7)
        self.x = self.x.reshape(self.x.shape[0], -1, 4)
        self.y = self.y.reshape(self.y.shape[0], -1)
        print('x shape:', self.x.shape)
        print('y shape:', self.y.shape)
        # self.max_x = np.max(self.x.reshape(-1, 7), axis=0)
        # self.min_x = np.min(self.x.reshape(-1, 7), axis=0)
        self.max_x = np.max(self.x.reshape(-1, 4), axis=0)
        self.min_x = np.min(self.x.reshape(-1, 4), axis=0)
        self.max_y = np.max(self.y.reshape(-1), axis=0)
        self.min_y = np.min(self.y.reshape(-1), axis=0)
        self.eps = 1e-12

    def normalize(self, x, y):
        x = (x - self.min_x) / (self.max_x - self.min_x + self.eps)
        y = (y - self.min_y) / (self.max_y - self.min_y + self.eps)
        return x, y

    def normalize_y(self, y):
        y = (y - self.min_y) / (self.max_y - self.min_y + self.eps)
        return y

    def denormalize(self, x, y):
        x = x * (self.max_x - self.min_x + self.eps) + self.min_x
        y = y * (self.max_y - self.min_y + self.eps) + self.min_y
        return x, y

    def denormalize_y(self, y):
        y = y * (self.max_y - self.min_y + self.eps) + self.min_y
        return y

    def __getitem__(self, index):
        return self.x[index, ...], self.y[index, ...]
    
    def __len__(self):
        return self.x.shape[0]
    
    def pointnumber(self):
        return self.x.shape[1]

class Darcyflow_2d_test():
    def __init__(self, root, r, nsample):
        self.dataset = scipy.io.loadmat(root)
        h = int(((421-1)/r)+1)
        s = h
        self.x = self.dataset['coeff'][-nsample:, ::r, ::r][:, :s, :s][..., None]
        # self.x_smooth = self.dataset['Kcoeff'][-nsample:, ::r, ::r][:, :s, :s][..., None]
        # self.x_gradx = self.dataset['Kcoeff_x'][-nsample:, ::r, ::r][:, :s, :s][..., None]
        # self.x_grady = self.dataset['Kcoeff_y'][-nsample:, ::r, ::r][:, :s, :s][..., None]
        # self.x = np.concatenate((self.x, self.x_smooth, self.x_gradx, self.x_grady), -1)
        self.y = self.dataset['sol'][-nsample:, ::r, ::r][:, :s, :s][..., None]
        grids = []
        grids.append(np.linspace(0, 1, s))
        grids.append(np.linspace(0, 1, s))
        grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
        grid = grid.reshape(1, s, s, 2)
        grid_add3 = np.zeros_like(grid)
        grid = np.concatenate((grid, grid_add3), -1)[..., :-1]
        self.x = np.concatenate((grid.repeat(nsample, 0), self.x), -1)
        # self.x = self.x.reshape(self.x.shape[0], -1, 7)
        self.x = self.x.reshape(self.x.shape[0], -1, 4)
        self.y = self.y.reshape(self.y.shape[0], -1)
        print('x shape:', self.x.shape)
        print('y shape:', self.y.shape)

    def __getitem__(self, index):
        return self.x[index, ...], self.y[index, ...]
    
    def __len__(self):
        return self.x.shape[0]
    
    def pointnumber(self):
        return self.x.shape[1]

if __name__=='__main__':

    path = '/home/data/dataset/hning/small_piececonst_r421_N1024_smooth2.mat'
    dataset = Darcyflow_2d(path, 5, 100)
    # x, y = dataset[2,6,30]
    # print(x[:5,100:105,:])
    # print(y[:5,100:105])
    # # print(x.shape)
    # # print(len(dataset))
    # # print(x[0,1000:1005,:])
    # # print(dataset.max_x, dataset.min_x, dataset.max_y, dataset.min_y)
    # x_1, y_1 = dataset.normalize(x, y)
    # print(x_1[:5,100:105,:])
    # print(y_1[:5,100:105])
    # x_2, y_2 = dataset.denormalize(x_1, y_1)
    # y_3 = dataset.denormalize_y(y_1)
    # print(x_2[:5,100:105,:])
    # print(y_2[:5,100:105])
    # print(y_3[:5,100:105])
    # # print(dataset.pointnumber())

