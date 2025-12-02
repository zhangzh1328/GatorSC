import os
import torch
import random
import numpy as np
from sklearn.metrics.cluster import *
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.optimize import linear_sum_assignment as linear_assignment


class CellDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.array((ind[0], ind[1])).T

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def evaluate(y_true, y_pred):
    acc= cluster_acc(y_true, y_pred)
    f1=0
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    homo = homogeneity_score(y_true, y_pred)
    comp = completeness_score(y_true, y_pred)
    return acc, f1, nmi, ari, homo, comp


def l1_distance(imputed_data, original_data):

    return np.mean(np.abs(original_data-imputed_data))


def pearson_corr(imputed_data, original_data):
    Y = original_data
    fake_Y = imputed_data
    fake_Y, Y = fake_Y.reshape(-1), Y.reshape(-1)
    fake_Y_mean, Y_mean = np.mean(fake_Y), np.mean(Y)
    corr = (np.sum((fake_Y - fake_Y_mean) * (Y - Y_mean))) / (
            np.sqrt(np.sum((fake_Y - fake_Y_mean) ** 2)) * np.sqrt(np.sum((Y - Y_mean) ** 2)))
    return corr


def evaluate_multiclass_average(all_y_true, all_y_pred):
    acc_list = []
    f1_list = []
    mcc_list = []

    for y_true, y_pred in zip(all_y_true, all_y_pred):
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        mcc = matthews_corrcoef(y_true, y_pred)

        acc_list.append(acc)
        f1_list.append(f1)
        mcc_list.append(mcc)

    metrics = {
        "accuracy": np.mean(acc_list),
        "f1": np.mean(f1_list),
        "mcc": np.mean(mcc_list)
    }
    
    return metrics


def predict_in_batches(model, z_batches, device):
    model.eval()
    preds = []
    for i in range(len(z_batches)):
        z = torch.tensor(z_batches[i]).to(device)
        logits = model(z)
        y_hat = torch.argmax(logits, dim=1)
        preds.append(y_hat.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)

