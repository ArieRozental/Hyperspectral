## imports

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from scipy.io import loadmat
from sklearn import metrics
from torch import nn
import torch.nn.functional as F
# import wandb
from sklearn.cluster import KMeans
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
torch.manual_seed(1)
from tqdm import tqdm
import os
import h5py
from sklearn.covariance import LedoitWolf
import pandas as pd
import sys


#channels_arr = [1, 2 ,4 , 8 , 12, 16, 20, 32, 48, 64, 81]  # FIXME ORARIE original: X.shape[2] = channels = X.shape[2]
channels_arr = [81]
for channels in channels_arr:
    channels = 81
    def read_hs_files(HS_files, File_path):
        Arrays = {}
        num = 1

        # Get the path of the first file and verify it is a .mat file
        if HS_files:
            first_file = HS_files[0]
            first_file_path = os.path.join(File_path, first_file)
            if first_file.endswith('.mat'):
                try:
                    with h5py.File(first_file_path, 'r') as f:
                        print("First file path:", first_file_path)
                except Exception as e:
                    print("Error reading the first .mat file:", e)
            else:
                print("The first file is not a .mat file:", first_file)

        for file in HS_files:
            input_img = os.path.join(File_path, file)
            print("Reading File[%d/%d]: %s" % (num, len(HS_files), input_img))
            try:
                with h5py.File(input_img, 'r') as f:
                    for k, v in f.items():
                        Arrays[file[:-4]] = np.array(v)
                        # print(k)
                        # print(v.shape)
            except Exception as e:
                print("Error reading the file:", e)
            num += 1

        return Arrays

    # Example usage
    HS_files = ["0082.mat"]
    File_path = "/home/adirido/HS-SOD/hyperspectral"
    arrays = read_hs_files(HS_files, File_path)
    #print(arrays)

    first_key = HS_files[0][:-4]  # Strip the '.mat' extension
    X = arrays.get(first_key)
    if X is not None:
        X = X / X.mean()  # Normalize the data
        X = X[:, 200:400, 100:300]  # Slice the data
        X = X.transpose(1,2,0)
        print("X (shape) = ", X.shape)
    else:
        print("Data for the first file not found.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    target = np.load('/home/adirido/hyperspectral_env/plastic.npy')
    target = target[0:channels]  # FIXME  ORARIE  original: [0:81]

    # theta = theta.reshape(-1)
    # tot_size = len(theta)
    X = X[:, :, 0:channels]

    print("X.shape =",X.shape)

    d = channels

    # sample_cov = np.cov(X.reshape(-1, d).T)
    # scov = sqrtm(sample_cov)
    # K = len(np.unique(theta))

    target = target / np.mean(target) * 2

    X = torch.from_numpy(X).float()

    target = torch.from_numpy(target).float()

    alpha = 0.01

    target_location = 1 - (torch.rand(X.shape[0], X.shape[1], 1) > 0.001).float()
    center_mask = torch.zeros(X.shape[0], X.shape[1], 1)
    center_mask[20:-20, 20:-20] = 1
    target_location = target_location * center_mask
    alpha_t = target_location * alpha
    # X_with_targets = (1 - alpha_t) * X + alpha_t * target.reshape(1, 1, channels)
    X_with_targets = X + alpha_t * target.reshape(1, 1, channels)


    all_inds_in_center = torch.where(center_mask.flatten())[0]
    all_inds_in_center = all_inds_in_center[torch.randperm(len(all_inds_in_center))]
    train_inds, test_inds = all_inds_in_center[:int(len(all_inds_in_center) * 0.8)], all_inds_in_center[
                                                                                     int(len(all_inds_in_center) * 0.8):]
    print("X with target (shape) = ", X_with_targets.shape)
    print("X_with_targets.numpy().shape", X_with_targets.numpy().shape)
    print("X_with_targets.numpy().reshape(-1, d) (shape) = ", X_with_targets.numpy().reshape(-1, d).shape)
    print("X_with_targets.numpy().reshape(-1, d).T (shape) = ", X_with_targets.numpy().reshape(-1, d).T.shape)
    sample_cov = np.cov(X_with_targets.numpy().reshape(-1, d).T)
    print("sample_cov (shape) = ", sample_cov.shape)
    #scov = sqrtm(sample_cov) # FIXME ORARIE original: scov = sqrtm(sample_cov) BXB=A
    #scov = torch.from_numpy(scov).float()
    # target = scov @target
    # X_with_targets = (scov[None,None] @ X_with_targets[:,:,:,None])[:,:,:,0]


    def flatten_tensor_by_distance(tensor):
        n = tensor.size(1)
        center = (n - 1) / 2  # Center position for odd-sized tensors

        # Create a tensor of indices
        indices = torch.cartesian_prod(torch.arange(n), torch.arange(n))

        # Calculate distances from the center for each index
        distances = torch.norm(indices - center, dim=1)

        # Sort indices based on distances
        _, sorted_indices = torch.sort(distances)

        # Flatten the tensor using the sorted indices for dimensions 1 and 2
        flattened_tensor = tensor.reshape(tensor.shape[0], -1)[:, sorted_indices]
        return flattened_tensor


    class MF:
        def __init__(self, name):
            self.name = name

        def __call__(self, x, target, alg='MF', use_cov=False):
            self.reconstruct(x)
            x_reconstructed = self.x_recunstructed

            x_reduced_mean = x - x_reconstructed

            if use_cov == 'before':
                cov = torch.cov(x.reshape(-1, channels).T)
            elif use_cov == 'after':
                cov = torch.cov(x_reduced_mean.reshape(-1, channels).T)
            else:
                cov = torch.eye(channels)

            inv_cov = torch.linalg.inv(cov)

            target = target.reshape(1, 1, 1, channels)
            inv_cov = inv_cov.reshape(1, 1, channels, channels)
            x_reduced_mean = x_reduced_mean.reshape(x.shape[0], x.shape[1], channels, 1)
            if alg == 'MF':
                score = target @ inv_cov @ x_reduced_mean
            elif alg == 'AMF' or alg == 'AMF_pos':
                score = (target @ inv_cov @ x_reduced_mean) **2 / (target @ inv_cov @ target.swapaxes(3, 2))
                if alg == 'AMF_pos':
                    score = score * torch.sign(target @ inv_cov @ x_reduced_mean)
            elif alg == 'ace' or alg == 'ace_pos':
                score = (target @ inv_cov @ x_reduced_mean) ** 2 / ((
                            x_reduced_mean.swapaxes(3, 2) @ inv_cov @ x_reduced_mean))
                if alg == 'ace_pos':
                    score = score * torch.sign(target @ inv_cov @ x_reduced_mean)
            score = score.reshape(x.shape[0], x.shape[1], 1)
            self.loglike = loglike(x, x_reconstructed, inv_cov,center_mask=center_mask)
            return score

        def calc_inv_cov(self, x):
            self.reconstruct(x)
            x_reconstructed = self.x_recunstructed
            x_reduced_mean = x - x_reconstructed
            cov = torch.cov(x_reduced_mean.reshape(-1, channels).T)
            inv_cov = torch.linalg.inv(cov)

            return inv_cov

        def reconstruct(self, x):
            raise NotImplementedError


    class MFKmeans(MF):

        def reconstruct(self, x):
            kmeans = KMeans(n_clusters=10, random_state=0).fit(x.reshape(-1, channels))

            clusters = kmeans.predict(x.reshape(-1, channels))

            clusters = clusters.reshape(X_with_targets.shape[0], X_with_targets.shape[1])
            clusters = torch.from_numpy(clusters).long()

            means = torch.from_numpy(kmeans.cluster_centers_).float()

            means_k = torch.zeros_like(X_with_targets)
            for i in range(len(means)):
                means_k[clusters == i] = means[i]
            self.x_recunstructed = means_k
            self.rec_score = rec_score(x, self.x_recunstructed)

            return means_k


    def rec_score(x, x_recunstructed):
        score = torch.mean((x_recunstructed - x) ** 2 * center_mask)
        return score


    class MFGlobal(MF):
        def __init__(self, name, use_cov=False):
            self.name = name
            self.use_cov = use_cov

        def reconstruct(self, x):
            mean = torch.mean(x, (0, 1))
            x_recunstructed = torch.ones_like(x) * mean[None, None]
            self.x_recunstructed = x_recunstructed
            self.rec_score = rec_score(x, x_recunstructed)


    class MFlocalMean(MF):
        def __init__(self, name, model='NULL', half_window_size=5, use_cov=False):
            self.name = name
            self.model = model
            self.use_cov = use_cov
            self.half_window_size = half_window_size
            self.kernel = torch.ones(1, 1, 1, half_window_size * 2 + 1, half_window_size * 2 + 1) / (
                        (half_window_size * 2 + 1) ** 2 - 1)
            self.kernel[:, :, :, half_window_size, half_window_size] = 0

        def reconstruct(self, x):
            x = x[None, None]
            x = x.permute(0, 1, 4, 2, 3)
            local_mean = F.conv3d(x, self.kernel, padding=(0, self.half_window_size, self.half_window_size))
            x_recunstructed = local_mean[0, 0].permute(1, 2, 0)
            x = x[0, 0].permute(1, 2, 0)
            self.x_recunstructed = x_recunstructed
            self.rec_score = rec_score(x, x_recunstructed)


    def generate_data_single(x, inds=None):
        half_window_size = 5
        if inds == 'train':
            flatten_inds = np.random.choice(train_inds, 1)[0]
        elif inds =='test':
            flatten_inds = np.random.choice(test_inds, 1)[0]
        else:
            flatten_inds = inds
        center_ind_x, center_ind_y = torch.div(flatten_inds, x.shape[1], rounding_mode='floor') , flatten_inds % x.shape[1]
        if center_ind_x < half_window_size + 1 or center_ind_x > x.shape[
            0] - half_window_size - 1 or center_ind_y < half_window_size + 1 or center_ind_y > x.shape[
            1] - half_window_size - 1:
            x_local_flatten = torch.zeros((2 * half_window_size + 1, 2 * half_window_size + 1, channels)).reshape(-1,
                                                                                                                  channels)
            y = torch.zeros(channels)
        else:
            inds_x = center_ind_x + torch.arange(-half_window_size, half_window_size + 1)
            inds_y = center_ind_y + torch.arange(-half_window_size, half_window_size + 1)
            x_local = x[inds_x][:, inds_y]
            x_local_flatten = flatten_tensor_by_distance(x_local.permute(2, 0, 1)).permute(1, 0)
            y = x_local_flatten[0]
        return x_local_flatten, y


    def generate_data_batch(x, batch_size, inds=None):
        xs, ys = zip(*[generate_data_single(x, inds=inds) for _ in range(batch_size)])
        xs = torch.stack(xs, 0)
        ys = torch.stack(ys, 0)
        return xs, ys


    def generate_data_batch_inds(x, all_inds):
        xs, ys = zip(*[generate_data_single(x, inds=inds) for inds in all_inds])
        xs = torch.stack(xs, 0)
        ys = torch.stack(ys, 0)
        return xs, ys

    class MFNet(MF):
        def __init__(self, name, model):
            self.name = name
            self.model = model
            self.x_recunstructed = None

        def reconstruct(self, x):
            if self.x_recunstructed is not None:
                return
            inds = torch.chunk(torch.arange(len(x.reshape(-1, channels))), 500)
            # Net_mean = torch.zeros_like(x).reshape(-1, channels)
            Net_mean = []
            for ind in tqdm(inds):
                xi, yi = generate_data_batch_inds(x, ind)
                Net_mean.append(self.model(xi.to(device))[0].cpu().detach())
            Net_mean = torch.cat(Net_mean, 0)
            Net_mean = Net_mean.reshape(x.shape)
            x_recunstructed = Net_mean
            self.x_recunstructed = x_recunstructed
            self.rec_score = rec_score(x, x_recunstructed)


    class MFNetCov(MF):
        def __init__(self, name, model):
            self.name = name
            self.model = model

        def __call__(self, x, target, alg='MF', use_cov=False):
            inds = torch.chunk(torch.arange(len(x.reshape(-1, channels))), 500)
            # Net_mean = torch.zeros_like(x).reshape(-1, channels)
            score = []
            rec_err_i = 0
            log_like_i = 0
            for i, ind in tqdm(enumerate(inds)):
                xi, yi = generate_data_batch_inds(x, ind)
                mean, cov = self.model(xi.to(device))
                mean, cov = mean.cpu().detach(), cov.cpu().detach()
                score_i, rec_err_i, log_like_i = self.detect_batch(yi, mean, cov, target,alg=alg)
                score.append(score_i)
                rec_err_i += rec_err_i * len(ind)
                log_like_i += log_like_i * len(ind)
            score = torch.cat(score, 0)
            self.rec_score = rec_err_i / len(score)
            self.loglike = log_like_i / len(score)
            score = score.reshape(x.shape[0], x.shape[1], 1)
            return score

        def detect_batch(self, x, x_reconstructed, inv_cov, target, alg='MF'):
            x_reduced_mean = x - x_reconstructed
            target = target.reshape(1, 1, channels)
            inv_cov = inv_cov.reshape(-1, channels, channels)
            x_reduced_mean = x_reduced_mean.reshape(-1, channels, 1) # Its all the image pixels in a row, and every pixel is a vector
            if alg == 'MF': # with zero mean.
                score = target @ inv_cov @ x_reduced_mean
            elif alg == 'AMF' or alg == 'AMF_pos':
                score = (target @ inv_cov @ x_reduced_mean) **2 / (target @ inv_cov @ target.swapaxes(2, 1))
                if alg == 'AMF_pos':
                    score = score * torch.sign(target @ inv_cov @ x_reduced_mean)
            elif alg == 'ace' or alg == 'ace_pos':
                score = (target @ inv_cov @ x_reduced_mean) ** 2 / ((
                            x_reduced_mean.swapaxes(2, 1) @ inv_cov @ x_reduced_mean))
                if alg == 'ace_pos':
                    score = score * torch.sign(target @ inv_cov @ x_reduced_mean)
            loglike_i = loglike(x, x_reconstructed, inv_cov)
            rec_score_i = loss_fn(x, x_reconstructed)
            return score, rec_score_i, loglike_i


    hidden_size = 100
    hidden_K = 10

    hidden_size_encoder = 20
    v_size = 100

    hidden_conv = 20


    class ConvNet(nn.Module):
        def __init__(self, output_size=100):
            super().__init__()
            self.conv1 = nn.Conv1d(1, hidden_conv, 3)
            self.conv2 = nn.Conv1d(hidden_conv, hidden_conv, 3, stride=2)
            self.conv3 = nn.Conv1d(hidden_conv, hidden_conv, 3)
            self.fc1 = nn.Linear(960, output_size)
            self.relu = nn.ReLU()

        def forward(self, x):
            org_shape = x.shape
            x = x.reshape(org_shape[0], 1, -1)
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.relu(x)
            x = x.reshape(org_shape[0], -1)
            x = self.fc1(x)
            x = x.reshape(-1)
            return x


    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            # self.means = torch.nn.Parameter(torch.randn(hidden_K, d))
            # self.means.requires_grad = True
            self.fc_encoder = nn.Linear(channels, hidden_size_encoder)
            # self.fc_encoder = ConvNet(output_size=hidden_size_encoder)
            self.fc_encoder_v = nn.Linear(channels, v_size)
            # self.fc_encoder_v = ConvNet(output_size=v_size)
            self.fc_decoder = nn.Linear(v_size, channels, bias=False)
            # self.fc_decoder.weight.data = means
            self.relu = nn.ReLU()
            self.sinv_cov_decoder = nn.Linear(v_size, channels **2, bias=True)
            self.sinvcov = torch.nn.Parameter(torch.randn(1, d, d))
            self.sinvcov.requires_grad = True

        def forward(self, x):
            batch_size, window_size, d = x.shape
            x_flatten = x.reshape(-1, d)
            encoded_x = self.fc_encoder(x_flatten)
            encoded_xv = self.fc_encoder_v(x_flatten)
            encoded_x = encoded_x.reshape(batch_size, window_size, hidden_size_encoder)
            encoded_xv = encoded_xv.reshape(batch_size, window_size, v_size)
            encoded_x_center = encoded_x[:, 1:5, :]
            encoded_x_reference = encoded_x[:, 1:, :]
            weights = encoded_x_center @ encoded_x_reference.swapaxes(2, 1)
            weights = nn.Softmax(2)(weights)
            weights = weights.swapaxes(2, 1)
            total_encoded = torch.sum(weights[:, :, None] * encoded_xv[:, 1:, :, None], 1)
            total_encoded = torch.mean(total_encoded, -1)
            decoded = self.fc_decoder(total_encoded)
            sinv_cov = self.sinv_cov_decoder(total_encoded)
            sinv_cov = sinv_cov.reshape(batch_size, channels, channels)
            inv_cov_local = sinv_cov @ sinv_cov.swapaxes(2,1)
            inv_cov_global = self.sinvcov @ self.sinvcov.swapaxes(2, 1) + 1e-6 * torch.eye(d).to(device)
            inv_cov = 0.8 * inv_cov_global + 0.2 * inv_cov_local
            # inv_cov = self.sinvcov @ self.sinvcov.swapaxes(2, 1) + 1e-6 * torch.eye(d).to(device)
            return decoded, inv_cov_local


    class New_Net(nn.Module):
        def __init__(self, name, model):
            super().__init__()
            self.name = name
            self.model = model

        def __call__(self, x, target, alg='MF', use_cov=True):
            # self.means = torch.nn.Parameter(torch.randn(hidden_K, d))
            # self.means.requires_grad = True
            self.fc_encoder = nn.Linear(channels, hidden_size_encoder)
            # self.fc_encoder = ConvNet(output_size=hidden_size_encoder)
            self.fc_encoder_v = nn.Linear(channels, v_size)
            # self.fc_encoder_v = ConvNet(output_size=v_size)
            self.fc_decoder = nn.Linear(v_size, channels, bias=False)
            # self.fc_decoder.weight.data = means
            self.relu = nn.ReLU()
            self.sinv_cov_decoder = nn.Linear(v_size, channels **2, bias=True)
            self.sinvcov = torch.nn.Parameter(torch.randn(1, d, d))
            self.sinvcov.requires_grad = True

        # def forward(self, x):
        #     batch_size, window_size, d = x.shape
        #     x_flatten = x.reshape(-1 , d)
        #     encoded_x = self.fc_encoder(x_flatten)
        #     encoded_xv = self.fc_encoder_v(x_flatten)
        #     encoded_x = encoded_x.reshape(batch_size, window_size, hidden_size_encoder)
        #     encoded_xv = encoded_xv.reshape(batch_size, window_size, v_size)
        #     encoded_x_center = encoded_x[:,0:1, :]
        #     encoded_x_reference = encoded_x[:,1:,:]
        #     weights = encoded_x_center @ encoded_x_reference.swapaxes(2,1)
        #     weights = nn.Softmax(2)(weights)
        #     weights = weights.swapaxes(2,1)
        #     total_encoded = torch.sum(encoded_xv[:,1:] * weights, 1)
        #     decoded = self.fc_decoder(total_encoded)

        #     return decoded
        def forward(self, x):
            batch_size, window_size, d = x.shape
            x_flatten = x.reshape(-1, d)
            encoded_x = self.fc_encoder(x_flatten)
            encoded_xv = self.fc_encoder_v(x_flatten)
            encoded_x = encoded_x.reshape(batch_size, window_size, hidden_size_encoder)
            encoded_xv = encoded_xv.reshape(batch_size, window_size, v_size)
            encoded_x_center = encoded_x[:, 1:5, :]
            encoded_x_reference = encoded_x[:, 1:, :]
            weights = encoded_x_center @ encoded_x_reference.swapaxes(2, 1)
            weights = nn.Softmax(2)(weights)
            weights = weights.swapaxes(2, 1)
            total_encoded = torch.sum(weights[:, :, None] * encoded_xv[:, 1:, :, None], 1)
            total_encoded = torch.mean(total_encoded, -1)
            decoded = self.fc_decoder(total_encoded)
            sinv_cov = self.sinv_cov_decoder(total_encoded)
            sinv_cov = sinv_cov.reshape(batch_size, channels, channels)
            inv_cov_local = sinv_cov @ sinv_cov.swapaxes(2,1) #+ torch.eye(d).to(device)
            inv_cov_global = self.sinvcov @ self.sinvcov.swapaxes(2, 1) + 1e-6 * torch.eye(d).to(device)
            return decoded, inv_cov_local

    def loss_fn(y, yhat):
        return torch.mean((y - yhat) ** 2)


    def loglike(y, yhat, inv_cov, center_mask=None):
        y = y.reshape(-1, channels)
        yhat = yhat.reshape(-1, channels)
        if center_mask is not None:
          center_mask = center_mask.reshape(-1)
          y = y[center_mask==1]
          yhat = yhat[center_mask==1]
        inv_cov = inv_cov.reshape(-1, channels, channels)
        return torch.mean(-(y - yhat)[:, None, :] @ inv_cov @ (y - yhat)[:, :, None] + torch.logdet(inv_cov))


    net = Net().to(device)
    net_cov = Net().to(device)
    net_local_cov = Net().to(device)  # New_net is same as Net only outputs local_inv_cov

    test_size = 100

    x_test, y_test = generate_data_batch(X_with_targets, test_size, inds='test')

    lr = 1e-4

    optimizer = torch.optim.Adam(lr=lr, params=net_cov.parameters())
    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, factor=0.5)

    iters = 50000 # change back to 50,000
    # iters = 0
    log_iter = 1000
    batch_size = 1

    lossses = []

    stop = False

    for it in range(iters):
        optimizer.zero_grad()
        x, y = generate_data_batch(X_with_targets, batch_size, inds='train')
        yhat, inv_cov = net_cov(x.to(device))
        loss = -loglike(yhat, y.to(device), inv_cov)
        loss.backward()
        optimizer.step()
        lossses.append(loss.item())
        if np.mod(it, log_iter) == 0:
            yhat_test, inv_cov_test = net_cov(x_test.to(device))
            loss_test = - loglike(yhat_test, y_test.to(device), inv_cov_test).item()
            rec_err = loss_fn(yhat_test, y_test.to(device)).item()

            loss_train = np.mean(lossses)
            schedular.step(loss_train)
            print(it, loss_train, loss_test, rec_err)
            lossses = []

        if stop:
            break

    lr = 1e-4

    optimizer = torch.optim.Adam(lr=lr, params=net.parameters())
    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, factor=0.5)

    iters = 0
    # iters = 0
    log_iter = 1000
    batch_size = 1

    lossses = []

    stop = False

    for it in range(iters):
        optimizer.zero_grad()
        x, y = generate_data_batch(X_with_targets, batch_size, inds='train')
        yhat, _ = net(x.to(device))
        loss = loss_fn(yhat, y.to(device))
        loss.backward()
        optimizer.step()
        lossses.append(loss.item())
        if np.mod(it, log_iter) == 0:
            yhat_test, _ = net(x_test.to(device))
            loss_test = loss_fn(yhat_test, y_test.to(device)).item()
            loss_train = np.mean(lossses)
            schedular.step(loss_train)
            print(it, loss_train, loss_test)
            lossses = []

        if stop:
            break

    fpr = {}
    tpr = {}
    tr = {}
    yhat = {}
    rec_scores = {}
    loglikes = {}

    # mf_global_cov = MFGlobal('mf_global_cov')
    mf_local_mean_global_cov = MFlocalMean('mf_local_mean_global_cov', half_window_size=10)
    # mf_kmeans = MFKmeans('mf_kemans')
    mf_net = MFNet('mf_net', model=net)
    mf_net_local_cov = MFNetCov('mf_net_local_cov', model=net_cov)

    models = [mf_net_local_cov]

    use_covs = ['after']
    # use_covs = ['after']
    algs = ['AMF_pos']

    for model in models:
        model_name = model.name
        for alg in algs:
            for use_cov in use_covs:
                #act_str = '_' + use_cov + '_' + alg
                # act_str = '_' + use_cov
                act_str =  "{:3d}_channels".format(channels)
                name = model_name + act_str
                yhat[name] = model(X_with_targets, target, use_cov=use_cov, alg=alg)
                yhat[name] -= torch.min(yhat[name]) + 1
                yhat[name] *= center_mask
                fpr[name], tpr[name], tr[name] = metrics.roc_curve(target_location.flatten().numpy(),
                                                                   yhat[name].flatten().numpy())
                plt.plot(fpr[name], tpr[name], label=name)
                plt.xscale('log')
                rec_scores[model_name] = model.rec_score.item()
                loglikes[model_name] = model.loglike.item()
                plt.xlabel('FPR')
                plt.ylabel('TPR')
                # print(model_name, 'rec_score', model.rec_score.item())
                print(model_name +'_' +use_cov, 'loglike', model.loglike.item())

    plt.legend()
    plt.show()
    print(rec_scores)
    inv_cov_net =  mf_net.calc_inv_cov(X_with_targets)
    yhat_test, _ = net(x_test.to(device))
    print(loglike(yhat_test, y_test.to(device), inv_cov_net.to(device)).item())