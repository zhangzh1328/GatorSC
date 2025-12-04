import argparse
import warnings
from sklearn.cluster import KMeans
import time
import json
import numpy as np
import scanpy as sc
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import *
from GatorSC import Model, ImputationRegressor, CellTypeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def train(train_loader,
          test_loader,
          input_dim, 
          num_head,
          hidden_dim, 
          mlp_dim, 
          num_hop, 
          phi_1, 
          phi_2, 
          phi_3, 
          del_rate, 
          add_rate, 
          max_path_len,
          tau,
          alpha_1,
          alpha_2,
          alpha_3,
          beta_1,
          beta_2,
          gamma_1,
          gamma_2,
          lambda_1,
          dropout,
          lr,
          seed,
          epochs,
          n_clusters,
          save_model_name,
          device):
    model = Model(input_dim, num_head, hidden_dim, mlp_dim, num_hop, phi_1, phi_2, phi_3, del_rate, add_rate, 
                  max_path_len, tau, alpha_1, alpha_2, alpha_3, beta_1, beta_2, gamma_1, gamma_2, lambda_1, dropout).to(device)

    opt_model = torch.optim.Adam(model.parameters(), lr=lr)

    setup_seed(seed)
    train_loss = []
    valid_loss = []
    min_loss = 999999
    best_epoch = 0
    
    print("="*15+" Self-supervised Training "+"="*15)
    start_time = time.time()
    for each_epoch in range(epochs):
        model.train()
        batch_loss = []

        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.float().to(device)

            batch_z, loss = model(batch_x)
            
            opt_model.zero_grad()
            loss.backward()
            opt_model.step()

            batch_loss.append(loss.cpu().detach().numpy())

        train_loss.append(np.mean(np.array(batch_loss)))
        
        epoch_time = time.time() - start_time
        avg_epoch_time = epoch_time / (each_epoch + 1)
        epochs_left = epochs - (each_epoch + 1)
        est_time_left = epochs_left * avg_epoch_time
        
        print(f"[Epoch {each_epoch + 1}/{epochs}] "
              f"Current train_loss: {train_loss[-1]:.4f} "
              f"Elapsed: {epoch_time/60:.2f} min "
              f"ETA: {est_time_left/60:.2f} min "
              f"({epochs_left} epochs left)")
        
        with torch.no_grad():
            model.eval()
            batch_loss = []
            
            for step, (batch_x, batch_y) in enumerate(valid_loader):
                batch_x = batch_x.float().to(device)

                batch_z, loss = model(batch_x)
                
                batch_loss.append(loss.cpu().detach().numpy())
        
        valid_loss.append(np.mean(np.array(batch_loss)))
        cur_loss = valid_loss[-1]
        if cur_loss < min_loss:
            min_loss = cur_loss
            best_epoch = each_epoch
            state = {
                    'net': model.state_dict(),
                    'optimizer': opt_model.state_dict(),
                    'epoch': best_epoch
                }
            torch.save(state, './saved_models/'+save_model_name+'_'+str(int(seed))+'_dict')

    return min_loss


def train_imputation(train_loader,
          valid_loader,
          input_dim, 
          num_head,
          hidden_dim, 
          mlp_dim, 
          num_hop, 
          phi_1, 
          phi_2, 
          phi_3, 
          del_rate, 
          add_rate, 
          max_path_len,
          tau,
          alpha_1,
          alpha_2,
          alpha_3,
          beta_1,
          beta_2,
          gamma_1,
          gamma_2,
          lambda_1,
          dropout,
          lr,
          seed,
          epochs,
          save_model_name,
          device):
    model = Model(input_dim, num_head, hidden_dim, mlp_dim, num_hop, phi_1, phi_2, phi_3, del_rate, add_rate, 
                  max_path_len, tau, alpha_1, alpha_2, alpha_3, beta_1, beta_2, gamma_1, gamma_2, lambda_1, dropout).to(device)
    ckpt_path = './saved_models/'+save_model_name+'_'+str(int(seed))+'_dict'
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['net'])
    model_imputation = ImputationRegressor(mlp_dim, input_dim).to(device)
    opt_model = torch.optim.Adam(model_imputation.parameters(), lr=lr)
    
    setup_seed(seed)
    train_loss = []
    valid_loss = []
    min_loss = 999999
    best_epoch = 0
    
    print("="*15 + " Imputation Task Training " + "="*15)
    start_time = time.time()
    for each_epoch in range(epochs):
        model.eval()
        model_imputation.train()
        batch_loss = []

        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.float().to(device)

            batch_z, _ = model(batch_x)
            x_hat = model_imputation(batch_z)
            
            mask = torch.where(batch_x != 0, torch.ones(batch_x.shape).to(device),
                               torch.zeros(batch_x.shape).to(device))
            mae_f = torch.nn.L1Loss(reduction='mean')
            loss = mae_f(mask * x_hat, mask * batch_x)

            opt_model.zero_grad()
            loss.backward()
            opt_model.step()

            batch_loss.append(loss.cpu().detach().numpy())

        train_loss.append(np.mean(np.array(batch_loss)))
        
        epoch_time = time.time() - start_time
        avg_epoch_time = epoch_time / (each_epoch + 1)
        epochs_left = epochs - (each_epoch + 1)
        est_time_left = epochs_left * avg_epoch_time

        print(f"[Epoch {each_epoch + 1}/{epochs}] "
              f"Current train_loss: {train_loss[-1]:.4f} "
              f"Elapsed: {epoch_time/60:.2f} min "
              f"ETA: {est_time_left/60:.2f} min "
              f"({epochs_left} epochs left)")
        
        with torch.no_grad():
            model_imputation.eval()
            batch_loss = []
            
            for step, (batch_x, batch_y) in enumerate(valid_loader):
                batch_x = batch_x.float().to(device)

                batch_z, _ = model(batch_x)
                x_hat = model_imputation(batch_z)

                mask = torch.where(batch_x != 0, torch.ones(batch_x.shape).to(device),
                                   torch.zeros(batch_x.shape).to(device))
                loss = mae_f(mask * x_hat, mask * batch_x)
                
                batch_loss.append(loss.cpu().detach().numpy())
        
        valid_loss.append(np.mean(np.array(batch_loss)))
        cur_loss = valid_loss[-1]
        if cur_loss < min_loss:
            min_loss = cur_loss
            best_epoch = each_epoch
            state = {
                    'net': model_imputation.state_dict(),
                    'optimizer': opt_model.state_dict(),
                    'epoch': best_epoch
                }
            torch.save(state, './saved_models/'+save_model_name+'_imputation_'+str(int(seed))+'_dict')

    return min_loss


def train_annotation(train_loader,
          valid_loader,
          input_dim, 
          num_head,
          hidden_dim, 
          mlp_dim, 
          num_hop, 
          phi_1, 
          phi_2, 
          phi_3, 
          del_rate, 
          add_rate, 
          max_path_len,
          tau,
          alpha_1,
          alpha_2,
          alpha_3,
          beta_1,
          beta_2,
          gamma_1,
          gamma_2,
          lambda_1,
          dropout,
          lr,
          seed,
          epochs,
          n_clusters,
          save_model_name,
          device):
    model = Model(input_dim, num_head, hidden_dim, mlp_dim, num_hop, phi_1, phi_2, phi_3, del_rate, add_rate, 
                  max_path_len, tau, alpha_1, alpha_2, alpha_3, beta_1, beta_2, gamma_1, gamma_2, lambda_1, dropout).to(device)
    ckpt_path = './saved_models/'+save_model_name+'_'+str(int(seed))+'_dict'
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['net'])
    model_annotation = CellTypeClassifier(mlp_dim, n_clusters).to(device)
    criterion = nn.CrossEntropyLoss()
    opt_model = torch.optim.Adam(model_annotation.parameters(), lr=lr)
    
    setup_seed(seed)
    train_loss = []
    valid_loss = []
    min_loss = 999999
    best_epoch = 0
    
    print("="*15 + " Cell Type Annotation Training " + "="*15)
    start_time = time.time()
    for each_epoch in range(epochs):
        model.eval()
        model_annotation.train()
        batch_loss = []

        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.long().to(device)
            
            batch_z, _ = model(batch_x)
            logits = model_annotation(batch_z)
            loss = criterion(logits, batch_y)
            
            opt_model.zero_grad()
            loss.backward()
            opt_model.step()

            batch_loss.append(loss.cpu().detach().numpy())

        train_loss.append(np.mean(np.array(batch_loss)))
        
        epoch_time = time.time() - start_time
        avg_epoch_time = epoch_time / (each_epoch + 1)
        epochs_left = epochs - (each_epoch + 1)
        est_time_left = epochs_left * avg_epoch_time

        print(f"[Epoch {each_epoch + 1}/{epochs}] "
              f"Current train_loss: {train_loss[-1]:.4f} "
              f"Elapsed: {epoch_time/60:.2f} min "
              f"ETA: {est_time_left/60:.2f} min "
              f"({epochs_left} epochs left)")
        
        with torch.no_grad():
            model_annotation.eval()
            batch_loss = []
            
            for step, (batch_x, batch_y) in enumerate(valid_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.long().to(device)
                
                batch_z, _ = model(batch_x)
                logits = model_annotation(batch_z)
                loss = criterion(logits, batch_y)
                
                batch_loss.append(loss.cpu().detach().numpy())
        
        valid_loss.append(np.mean(np.array(batch_loss)))
        cur_loss = valid_loss[-1]
        if cur_loss < min_loss:
            min_loss = cur_loss
            best_epoch = each_epoch
            state = {
                    'net': model_annotation.state_dict(),
                    'optimizer': opt_model.state_dict(),
                    'epoch': best_epoch
                }
            torch.save(state, './saved_models/'+save_model_name+'_annotation_'+str(int(seed))+'_dict')

    return min_loss


def test(test_loader,
          input_dim,
          num_head,
          hidden_dim, 
          mlp_dim, 
          num_hop, 
          phi_1, 
          phi_2, 
          phi_3, 
          del_rate, 
          add_rate, 
          max_path_len,
          tau,
          alpha_1,
          alpha_2,
          alpha_3,
          beta_1,
          beta_2,
          gamma_1,
          gamma_2,
          lambda_1,
          dropout,
          lr,
          seed,
          epochs,
          n_clusters,
          save_model_name,
          task,
          device):
    model = Model(input_dim, num_head, hidden_dim, mlp_dim, num_hop, phi_1, phi_2, phi_3, del_rate, add_rate, 
                  max_path_len, tau, alpha_1, alpha_2, alpha_3, beta_1, beta_2, gamma_1, gamma_2, lambda_1, dropout).to(device)
    ckpt_path = './saved_models/'+save_model_name+'_'+str(int(seed))+'_dict'
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['net'])
        
    z_test = []
    x_test = []
    y_test = []
    for step, (batch_x, batch_y) in enumerate(test_loader):
        batch_x = batch_x.float().to(device)

        batch_z, _ = model(batch_x)

        z_test.append(batch_z.cpu().detach().numpy())
        x_test.append(batch_x.cpu().detach().numpy())
        y_test.append(batch_y.detach().numpy())

    z_test_list = z_test
    z_test = np.vstack(z_test)
    x_test = np.vstack(x_test)
    y_test_list = y_test
    y_test = np.hstack(y_test)
    
    if task == 'clustering':
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit(z_test)
        y_kmeans_test = kmeans.labels_

        acc, _, nmi, ari, homo, _ = evaluate(y_test, y_kmeans_test)
        result = {"acc": acc, "nmi": nmi, "ari": ari, "homo": homo}

        return result
    elif task == 'imputation':
        min_loss = train_imputation(train_loader, valid_loader, input_dim, num_head, hidden_dim, mlp_dim, num_hop, phi_1, phi_2, phi_3, 
                                    del_rate, add_rate, max_path_len, tau, alpha_1, alpha_2, alpha_3, beta_1, beta_2, gamma_1, gamma_2, 
                                    lambda_1, dropout, lr, seed, epochs, save_model_name, device)
        model_imputation = ImputationRegressor(mlp_dim, input_dim).to(device)
        ckpt_path = './saved_models/'+save_model_name+'_imputation_'+str(int(seed))+'_dict'
        state = torch.load(ckpt_path, map_location=device)
        model_imputation.load_state_dict(state['net'])
        
        x_hat = model_imputation(torch.tensor(z_test).to(device)).cpu().detach().numpy()
        mask = np.where(x_test != 0, np.ones_like(x_test), np.zeros_like(x_test))
        
        pcc = pearson_corr(mask * x_hat, mask * x_test)
        l1 = l1_distance(mask * x_hat, mask * x_test)
        result = {"pcc": pcc, "l1": l1}
        
        return result 
    elif task == 'annotation':
        min_loss = train_annotation(train_loader, valid_loader, input_dim, num_head, hidden_dim, mlp_dim, num_hop, phi_1, phi_2, phi_3, 
                                    del_rate, add_rate, max_path_len, tau, alpha_1, alpha_2, alpha_3, beta_1, beta_2, gamma_1, gamma_2, 
                                    lambda_1, dropout, lr, seed, epochs, n_clusters, save_model_name, device)
        model_annotation = CellTypeClassifier(mlp_dim, n_clusters).to(device)
        ckpt_path = './saved_models/'+save_model_name+'_annotation_'+str(int(seed))+'_dict'
        state = torch.load(ckpt_path, map_location=device)
        model_annotation.load_state_dict(state['net'])
        
        y_test_true = np.concatenate([y_test_list[i] for i in range(len(y_test_list))], axis=0)
        y_test_pred = predict_in_batches(model_annotation, z_test_list, device)
                
        result = evaluate_multiclass_average([y_test_true], [y_test_pred])
        
        return result
       

if __name__ == '__main__':  
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, default="./data/")
    parser.add_argument("--data_name", type=str, default="BCs-PCs")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_head", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=225)
    parser.add_argument("--num_hop", type=int, default=2)
    parser.add_argument("--phi_1", type=float, default=0.05)
    parser.add_argument("--phi_2", type=float, default=0.02)
    parser.add_argument("--phi_3", type=float, default=0.03)
    parser.add_argument("--del_rate", type=float, default=0.4)
    parser.add_argument("--add_rate", type=float, default=0.4)
    parser.add_argument("--max_path_len", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.25)
    parser.add_argument("--alpha_1", type=float, default=0.55)
    parser.add_argument("--alpha_2", type=float, default=0.53)
    parser.add_argument("--alpha_3", type=float, default=0.62)
    parser.add_argument("--beta_1", type=float, default=0.9)
    parser.add_argument("--beta_2", type=float, default=0.85)
    parser.add_argument("--gamma_1", type=float, default=0.76)
    parser.add_argument("--gamma_2", type=float, default=0.89)
    parser.add_argument("--lambda_1", type=float, default=0.68)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save_model_name", type=str, default="model")
    parser.add_argument("--task", type=str, default="clustering", choices=["clustering", "imputation", "annotation"])
    
    args = parser.parse_args()
    data_name = args.data_name
    folder_path = args.folder_path
    device_n = args.device
    batch_size = args.batch_size
    num_head = args.num_head
    hidden_dim = args.hidden_dim
    mlp_dim = args.hidden_dim
    num_hop = args.num_hop
    phi_1 = args.phi_1
    phi_2 = args.phi_2
    phi_3 = args.phi_3
    del_rate = args.del_rate
    add_rate = args.add_rate
    max_path_len = args.max_path_len
    tau = args.tau
    alpha_1 = args.alpha_1
    alpha_2 = args.alpha_2
    alpha_3 = args.alpha_3
    beta_1 = args.beta_1
    beta_2 = args.beta_2
    gamma_1 = args.gamma_1
    gamma_2 = args.gamma_2
    lambda_1 = args.lambda_1
    dropout = args.dropout
    lr = args.lr
    seed = args.seed
    epochs = args.epochs
    save_model_name = data_name+'_'+args.save_model_name
    task = args.task
    
    device = torch.device(f"cuda:{device_n}")
    data_path = folder_path+data_name+'.h5ad'
    adata = sc.read_h5ad(data_path)
    try:
        X_all = adata.X.toarray()
    except:
        X_all = adata.X
    y_all = adata.obs.loc[:,'cell_type']
    input_dim = X_all.shape[1]
    
    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(y_all)
    n_clusters = len(np.unique(y_all))

    X_train, X_valid_test, y_train, y_valid_test = train_test_split(X_all, y_all, test_size=0.2, random_state=1)
    X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, test_size=0.5, random_state=1)

    train_set = CellDataset(X_train, y_train)
    valid_set = CellDataset(X_valid, y_valid)
    test_set = CellDataset(X_test, y_test)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=10)
    valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=False, num_workers=10)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=10)
    
    train_loss = train(train_loader, test_loader, input_dim, num_head, hidden_dim, mlp_dim, num_hop, phi_1, phi_2, phi_3, 
                       del_rate, add_rate, max_path_len, tau, alpha_1, alpha_2, alpha_3, beta_1, beta_2, gamma_1, gamma_2, 
                       lambda_1, dropout, lr, seed, epochs, n_clusters, save_model_name, device)
    result = test(test_loader, input_dim, num_head, hidden_dim, mlp_dim, num_hop, phi_1, phi_2, phi_3, 
                  del_rate, add_rate, max_path_len, tau, alpha_1, alpha_2, alpha_3, beta_1, beta_2, gamma_1, 
                  gamma_2, lambda_1, dropout, lr, seed, epochs, n_clusters, save_model_name, task, device)
    
    result = {k: float(v) for k, v in result.items()}
    
    with open(f"./saved_results/{data_name}_{task}_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f)
