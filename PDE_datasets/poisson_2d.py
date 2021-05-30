"""
Modified from PointConv: https://github.com/DylanWusee/pointconv
Author: Ning Hua
Date: May 2021
"""

import numpy as np

class Poisson_2d():
    def __init__(self, root, nsample):
        self.dataset = np.load(root)
        self.x = self.dataset[:nsample, :, :, 0][..., None]
        self.y = self.dataset[:nsample, :, :, 1][..., None]
        s = 80
        grids = []
        grids.append(np.linspace(-1, 1, s))
        grids.append(np.linspace(-1, 1, s))
        grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
        grid = grid.reshape(1, s, s, 2)
        grid_add3 = np.zeros_like(grid)
        grid = np.concatenate((grid, grid_add3), -1)[..., :-1]
        self.x = np.concatenate((grid.repeat(nsample, 0), self.x), -1)
        self.x = self.x.reshape(self.x.shape[0], -1, 4)
        self.y = self.y.reshape(self.y.shape[0], -1)
        print('x shape:', self.x.shape)
        print('y shape:', self.y.shape)
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

class Poisson_2d_test():
    def __init__(self, root, nsample):
        self.dataset = np.load(root)
        self.x = self.dataset[-nsample:, :, :, 0][..., None]
        self.y = self.dataset[-nsample:, :, :, 1][..., None]
        s = 80
        grids = []
        grids.append(np.linspace(-1, 1, s))
        grids.append(np.linspace(-1, 1, s))
        grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
        grid = grid.reshape(1, s, s, 2)
        grid_add3 = np.zeros_like(grid)
        grid = np.concatenate((grid, grid_add3), -1)[..., :-1]
        self.x = np.concatenate((grid.repeat(nsample, 0), self.x), -1)
        self.x = self.x.reshape(self.x.shape[0], -1, 4)
        self.y = self.y.reshape(self.y.shape[0], -1)
        print('x shape:', self.x.shape)
        print('y shape:', self.y.shape)
        self.max_x = np.max(self.x.reshape(-1, 4), axis=0)
        self.min_x = np.min(self.x.reshape(-1, 4), axis=0)
        self.max_y = np.max(self.y.reshape(-1), axis=0)
        self.min_y = np.min(self.y.reshape(-1), axis=0)
        self.eps = 1e-12

    def __getitem__(self, index):
        return self.x[index, ...], self.y[index, ...]
    
    def __len__(self):
        return self.x.shape[0]
    
    def pointnumber(self):
        return self.x.shape[1]

class Poisson_2d_downsample():
    def __init__(self, root):
        self.dataset = np.load(root)
        self.x = self.dataset[:, :, :4]
        self.y = self.dataset[:, :, -1]
        print('x shape:', self.x.shape)
        print('y shape:', self.y.shape)
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


class Poisson_2d_unstructed_generator():
    def __init__(self, root, nsample):
        self.dataset = np.load(root)
        self.x = self.dataset[:nsample, :, :, 0][..., None]
        self.y = self.dataset[:nsample, :, :, 1][..., None]
        s = 80
        grids = []
        grids.append(np.linspace(-1, 1, s))
        grids.append(np.linspace(-1, 1, s))
        grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
        grid = grid.reshape(1, s, s, 2)
        grid_add3 = np.zeros_like(grid)
        grid = np.concatenate((grid, grid_add3), -1)[..., :-1]
        self.x = np.concatenate((grid.repeat(nsample, 0), self.x), -1)
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
    
    def downsample(self):
        x_down = np.zeros((self.x.shape[0], 4000, self.x.shape[2]))
        y_down = np.zeros((self.y.shape[0], 4000))
        for i in range(self.x.shape[0]):
            index = np.random.choice(self.x.shape[1], 4000, replace=False)
            x_down[i, :, :] = self.x[i, index, :]
            y_down[i, :] = self.y[i, index]
        print('x_down shape: ', x_down.shape)
        print('y_down shape: ', y_down.shape)
        print('x_down slice: ', x_down[[0,50,-1], 3995:4000, :])
        print('y_down slice: ', y_down[[0,50,-1], 3995:4000])
        xy_down = np.concatenate((x_down, y_down[..., None]), -1)
        print('xy_down shape: ', xy_down.shape)
        np.save('/home/data/dataset/hning/poisson_2d_K3_down4000_tt.npy', xy_down)

if __name__=='__main__':

    path = '/home/data/dataset/hning/poisson_2d_K3_down4000_train.npy'

    ################# Poisson_2d_downsample #######################
    dataset = Poisson_2d_downsample(path)
    x, y = dataset[2,6,30]
    print(x[:5,100:105,:])
    print(y[:5,100:105])
    # print(x.shape)
    # print(len(dataset))
    # print(x[0,1000:1005,:])
    # print(dataset.max_x, dataset.min_x, dataset.max_y, dataset.min_y)
    x_1, y_1 = dataset.normalize(x, y)
    print(x_1[:5,100:105,:])
    print(y_1[:5,100:105])
    x_2, y_2 = dataset.denormalize(x_1, y_1)
    y_3 = dataset.denormalize_y(y_1)
    print(x_2[:5,100:105,:])
    print(y_2[:5,100:105])
    print(y_3[:5,100:105])
    # print(dataset.pointnumber())

    ################# Poisson_2d_unstructed_generator##############
    # dataset = Poisson_2d_unstructed_generator(path, 1000)
    # dataset.downsample()

    ################# Poisson_2d ################
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

