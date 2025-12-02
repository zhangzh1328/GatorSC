import torch
import torch.nn as nn
import copy
import math
from scipy.sparse import coo_matrix
import torch.nn.functional as F
import numpy as np
import random
from collections import defaultdict
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.conv import GCNConv
import networkx as nx
device = torch.device("cuda" if torch.cuda.is_available() == True else 'cpu')


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class GlobalCellGraph(nn.Module):
    def __init__(self, h, input_dim, hid_dim, phi, dropout=0):
        super().__init__()
        assert input_dim % h == 0

        self.d_k = input_dim // h
        self.h = h
        self.linears = clones(nn.Linear(input_dim, self.d_k * self.h), 2)
        self.dropout = nn.Dropout(p=dropout)
        self.Wo = nn.Linear(h, 1)
        self.phi = nn.Parameter(torch.tensor(phi), requires_grad=True)

    def forward(self, query, key):
        query, key = [l(x).view(query.size(0), -1, self.h, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]

        attns = self.attention(query.squeeze(2), key.squeeze(2))
        adj = torch.where(attns >= self.phi, torch.ones(attns.shape).to(device), torch.zeros(attns.shape).to(device))

        return adj

    def attention(self, query, key):
        d_k = query.size(-1)
        scores = torch.bmm(query.permute(1, 0, 2), key.permute(1, 2, 0)) \
                 / math.sqrt(d_k)
        scores = self.Wo(scores.permute(1, 2, 0)).squeeze(2)
        p_attn = F.softmax(scores, dim=1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return p_attn


def get_k_hop_subgraph_nodes(G, node, hop_number):
    visited = set([node])
    frontier = set([node])
    for _ in range(hop_number):
        next_frontier = set()
        for n in frontier:
            next_frontier.update(set(nx.neighbors(G, n)))
        next_frontier -= visited  
        visited.update(next_frontier)
        frontier = next_frontier
    return list(visited)


def SubGraph(X_all, G, hop_number):
    X_all = torch.tensor(X_all).float().to(device)
    
    max_neighbors = 10  
    subgraph_data_list = []

    for node in G.nodes:  
        subgraph_nodes = get_k_hop_subgraph_nodes(G, node, hop_number)
        total_nodes = [node] + subgraph_nodes
        total_nodes = list(set(total_nodes))[:max_neighbors]  
        subgraph_nodes = total_nodes
        subgraph = G.subgraph(subgraph_nodes)

        node_mapping = {n: i for i, n in enumerate(subgraph_nodes)}
        edge_index = torch.tensor(
            [[node_mapping[u], node_mapping[v]] for u, v in subgraph.edges], dtype=torch.long
        ).t().contiguous()

        x = X_all[subgraph_nodes]
        data = {'x': x, 'edge_index': edge_index.to(device)}
        subgraph_data_list.append(data)

    return subgraph_data_list
    

class GlobalGeneGraph(nn.Module):
    def __init__(self, h, input_dim, hid_dim, phi, dropout = 0):
        super().__init__()
        assert input_dim % h == 0
        
        self.w_h = nn.Linear(input_dim, hid_dim)
        input_dim = hid_dim
        self.d_k = input_dim // h
        self.h = h
        self.linears = clones(nn.Linear(input_dim, self.d_k * self.h), 2)
        self.dropout = nn.Dropout(p=dropout)
        self.Wo = nn.Linear(h, 1)
        self.phi = nn.Parameter(torch.tensor(phi), requires_grad=True)

    def forward(self, query, key):
        query = self.w_h(query)
        key = self.w_h(key)
        
        query, key = [l(x).view(query.size(0), -1, self.h, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        
        query = query.permute(3,1,2,0)
        key = key.permute(3,1,2,0)

        attns = self.attention(query.squeeze(2), key.squeeze(2))
        adj = torch.where(attns >= self.phi, torch.ones(attns.shape).to(device), torch.zeros(attns.shape).to(device))

        return adj

    def attention(self, query, key):
        d_k = query.size(-1)
        scores = torch.bmm(query.permute(1, 0, 2), key.permute(1, 2, 0)) \
                 / math.sqrt(d_k)
        scores = self.Wo(scores.permute(1, 2, 0)).squeeze(2)
        p_attn = F.softmax(scores, dim=1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return p_attn


class LocalGeneGraph(nn.Module):
    def __init__(self, h, input_dim, hid_dim, phi, pool_len=256, dropout=0.0, pool_type="avg"):
        super().__init__()
        assert hid_dim % h == 0
        assert pool_type in ["avg", "max"]

        self.w_h = nn.Linear(input_dim, hid_dim)
        self.h = h
        self.d_k = hid_dim // h
        self.pool_len = pool_len
        self.pool_type = pool_type

        self.q_proj = nn.Linear(pool_len, self.d_k * self.h)
        self.k_proj = nn.Linear(pool_len, self.d_k * self.h)

        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else None
        self.Wo = nn.Linear(h, 1)
        self.phi = nn.Parameter(torch.tensor(phi), requires_grad=True)

    def forward(self, subgraph_data_list):
        sub_gene_graph_list = []
        for i in range(len(subgraph_data_list)):
            node_feature = subgraph_data_list[i]['x']          
            device = node_feature.device

            node_hid = self.w_h(node_feature)                  

            feat_tokens = node_hid.transpose(0, 1)              

            feat_tokens = self.pool_to_fixed_len(feat_tokens)   

            attn = self.attention(feat_tokens, feat_tokens)     

            sub_adj = torch.where(
                attn >= self.phi,
                torch.ones_like(attn, device=device),
                torch.zeros_like(attn, device=device)
            )
            sub_gene_graph_list.append(sub_adj)

        return sub_gene_graph_list

    def pool_to_fixed_len(self, feat_tokens):
        x = feat_tokens.unsqueeze(0)  
        if self.pool_type == "avg":
            x = F.adaptive_avg_pool1d(x, self.pool_len)   
        else:
            x = F.adaptive_max_pool1d(x, self.pool_len)   
        return x.squeeze(0)  

    def attention(self, query, key):
        T = query.size(0)

        query = self.q_proj(query).view(T, self.h, self.d_k).transpose(0, 1)  
        key   = self.k_proj(key).view(T, self.h, self.d_k).transpose(0, 1)    

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)  

        scores = scores.permute(1, 2, 0)          
        scores = self.Wo(scores).squeeze(-1)      

        p_attn = F.softmax(scores, dim=1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return p_attn


class GlobalGeneGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(support, adj)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GlobalGeneGCN(nn.Module):
    def __init__(self, input_size, hid1_size, hid2_size):
        super().__init__()
        self.gc1 = GlobalGeneGraphConvolution(input_size, hid1_size)
        self.gc2 = GlobalGeneGraphConvolution(hid1_size, hid2_size)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class LocalGeneGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.bmm(support.unsqueeze(1), adj).squeeze()
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class LocalGeneGCN(nn.Module):
    def __init__(self, input_size, hid1_size, hid2_size):
        super().__init__()
        self.gc1 = LocalGeneGraphConvolution(input_size, hid1_size)
        self.gc2 = LocalGeneGraphConvolution(hid1_size, hid2_size)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


def add_del_path(edge_index, add_rate=0.0, del_rate=0.0, num_nodes=None, max_path_len=3, device='cpu'):
    edge_start, edge_end = edge_index[0].tolist(), edge_index[1].tolist()
    edges = set(zip(edge_start, edge_end))
    num_edges = len(edges)
    if num_nodes is None:
        num_nodes = max(max(edge_start), max(edge_end)) + 1

    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)

    del_num = int(del_rate * num_edges)
    to_del = set()
    n_deleted_path = 0
    attempts = 0
    max_attempts = max(del_num * 10, 50)
    while n_deleted_path < del_num and attempts < max_attempts:
        start = random.randint(0, num_nodes-1)
        path_len = random.randint(2, max_path_len)
        path = [start]
        cur = start
        for _ in range(path_len-1):
            if len(adj[cur]) == 0: break
            next_node = random.choice(adj[cur])
            path.append(next_node)
            cur = next_node
        if len(path) >= 2:
            path_edges = set(zip(path[:-1], path[1:]))
            if not path_edges & to_del and path_edges <= edges:
                to_del |= path_edges
                n_deleted_path += 1
        attempts += 1

    add_num = int(add_rate * num_edges)
    to_add = set()
    n_added_path = 0
    attempts = 0
    max_attempts = max(add_num * 10, 50)
    while n_added_path < add_num and attempts < max_attempts:
        path_len = random.randint(2, max_path_len)
        path = [random.randint(0, num_nodes-1)]
        for _ in range(path_len-1):
            next_node = random.randint(0, num_nodes-1)
            path.append(next_node)
        path_edges = set(zip(path[:-1], path[1:]))
        
        path_edges_to_add = {e for e in path_edges if e not in edges and e not in to_del and e not in to_add}
        if path_edges_to_add:
            to_add |= path_edges_to_add
            n_added_path += 1
        attempts += 1
    
    final_edges = (edges | to_add) - to_del
    if final_edges:
        edge_start_new, edge_end_new = zip(*final_edges)
    else:
        edge_start_new, edge_end_new = [], []
    new_edge_index = torch.tensor([edge_start_new, edge_end_new], dtype=torch.long, device=device)
    return new_edge_index


def reconstruction_loss(z, pos_edge_index, num_nodes, neg_sample_ratio=1.0):
    src, dst = pos_edge_index
    pos_score = (z[src] * z[dst]).sum(dim=1)     
    pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))

    num_neg = int(pos_edge_index.shape[1] * neg_sample_ratio)
    neg_src = torch.randint(0, num_nodes, (num_neg,))
    neg_dst = torch.randint(0, num_nodes, (num_neg,))
    
    neg_score = (z[neg_src] * z[neg_dst]).sum(dim=1)
    neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))

    loss = pos_loss + neg_loss
    
    return loss


def sim(z1, z2, hidden_norm):
    if hidden_norm:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
    return torch.mm(z1, z2.T)


def cl_loss(z, z_aug, adj, tau, hidden_norm=True):
    f = lambda x: torch.exp(x / tau)
    intra_view_sim = f(sim(z, z, hidden_norm))
    inter_view_sim = f(sim(z, z_aug, hidden_norm))

    positive = inter_view_sim.diag() + (intra_view_sim.mul(adj)).sum(1) + (inter_view_sim.mul(adj)).sum(1)

    loss = positive / (intra_view_sim.sum(1) + inter_view_sim.sum(1) - intra_view_sim.diag())

    adj_count = torch.sum(adj, 1) * 2 + 1
    loss = torch.log(loss) / adj_count

    return -torch.mean(loss, 0)


def final_cl_loss(alpha1, alpha2, z, z_aug, adj, adj_aug, tau, hidden_norm=True):
    loss = alpha1 * cl_loss(z, z_aug, adj, tau, hidden_norm) + alpha2 * cl_loss(z_aug, z, adj_aug, tau, hidden_norm)

    return loss


class Model(nn.Module):
    def __init__(self, input_dim, num_head, hidden_dim, mlp_dim, num_hop, phi_1, phi_2, phi_3, 
                 del_rate, add_rate, max_path_len, tau, a1, a2, b1, b2, b3, lambda_1, lambda_2, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_hop = num_hop
        self.del_rate = del_rate
        self.add_rate = add_rate
        self.max_path_len = max_path_len
        self.tau = tau
        self.a1 = a1
        self.a2 = a2
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        
        self.global_cell_graph = GlobalCellGraph(num_head, input_dim, hidden_dim, phi_1)  
        self.global_gene_graph = GlobalGeneGraph(num_head, input_dim, hidden_dim, phi_2)
        self.local_gene_graph = LocalGeneGraph(num_head, input_dim, hidden_dim, phi_3)
        
        self.global_cell_gcn = GCNConv(input_dim, hidden_dim)
        self.global_gene_gcn = GlobalGeneGCN(input_dim, hidden_dim, hidden_dim)
        self.local_gene_gcn = LocalGeneGCN(input_dim, hidden_dim, hidden_dim)
        
        self.global_cell_mlp = nn.Linear(hidden_dim, mlp_dim)
        self.global_gene_mlp = nn.Linear(hidden_dim, mlp_dim)
        self.local_gene_mlp = nn.Linear(hidden_dim, mlp_dim)
        
        self.rounter = nn.Linear(mlp_dim*3, 3)
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, input):
        input = self.dropout(input)
        
        global_cell_adj = self.global_cell_graph(input, input) 
        global_cell_adj_np = global_cell_adj.cpu().detach().numpy()
        G = nx.from_numpy_array(global_cell_adj_np)
        subgraph_data_list = SubGraph(input, G, self.num_hop)
        
        local_gene_adj = torch.stack(self.local_gene_graph(subgraph_data_list))
        
        global_gene_adj = self.global_gene_graph(input, input)
        
        global_cell_edge_index = torch.nonzero(global_cell_adj == 1).T
        global_cell_emb = self.global_cell_gcn(input, global_cell_edge_index)
        global_gene_emb = self.global_gene_gcn(input, global_gene_adj)
        local_gene_emb = self.local_gene_gcn(input, local_gene_adj)
        
        global_cell_emb= self.global_cell_mlp(global_cell_emb) 
        global_gene_emb = self.global_gene_mlp(F.normalize(global_gene_emb, p=2, dim=1))
        local_gene_emb = self.local_gene_mlp(F.normalize(local_gene_emb, p=2, dim=1))

        expert_logits= self.rounter(torch.cat((global_cell_emb, local_gene_emb, global_gene_emb), dim=1))
        weights = F.softmax(expert_logits, dim=1)
        combined_emb= weights[:, 0:1] * global_cell_emb + weights[:, 1:2] * local_gene_emb + weights[:, 2:3] * global_gene_emb
        
        global_cell_edge_index = torch.nonzero(global_cell_adj == 1).T
        num_nodes = maybe_num_nodes(global_cell_edge_index)
        global_cell_edge_index_path = add_del_path(global_cell_edge_index, add_rate=self.add_rate, del_rate=self.del_rate,
                             num_nodes=num_nodes, max_path_len=self.max_path_len, device=device)
        
        global_cell_edge_index_path_T = global_cell_edge_index_path.T
        global_cell_path_adj = torch.zeros(input.shape[0], input.shape[0]).to(device)
        global_cell_path_adj[global_cell_edge_index_path_T[:, 0], global_cell_edge_index_path_T[:, 1]] = 1
        
        global_cell_path_emb = self.global_cell_gcn(input, global_cell_edge_index_path)
        global_cell_path_emb= self.global_cell_mlp(global_cell_path_emb)  
        
        adj_hat = torch.mm(global_cell_path_emb, global_cell_path_emb.T)
        loss_dam = F.binary_cross_entropy(torch.sigmoid(adj_hat), global_cell_adj)

        loss_mgm_1 = loss_dam 
        
        local_gene_edge_index = torch.nonzero(local_gene_adj == 1).T
        
        loss_raw = reconstruction_loss(global_cell_emb, global_cell_edge_index, input.shape[0])  
        loss_aug = reconstruction_loss(global_cell_path_emb, global_cell_edge_index_path, input.shape[0])  
        loss_rec_1 = loss_raw + loss_aug 
        
        loss_cl_1 = final_cl_loss(self.a1, self.a2, global_cell_emb, global_cell_path_emb, global_cell_adj, global_cell_path_adj, tau=self.tau)
        
        
        global_gene_edge_index = torch.nonzero(global_gene_adj == 1).T
        num_nodes = maybe_num_nodes(global_gene_edge_index)
        global_gene_edge_index_path = add_del_path(global_gene_edge_index, add_rate=self.add_rate, del_rate=self.del_rate,
                             num_nodes=num_nodes, max_path_len=self.max_path_len, device=device)
        
        global_gene_edge_index_path_T = global_gene_edge_index_path.T
        global_gene_path_adj = torch.zeros(self.hidden_dim, self.hidden_dim).to(device)
        global_gene_path_adj[global_gene_edge_index_path_T[:, 0], global_gene_edge_index_path_T[:, 1]] = 1
        
        global_gene_path_emb = self.global_gene_gcn(input, global_gene_path_adj)
        global_gene_path_emb= self.global_gene_mlp(global_gene_path_emb)  
        adj_hat = torch.mm(global_gene_path_emb.T, global_gene_path_emb)
        
        loss_dam = F.binary_cross_entropy(torch.sigmoid(adj_hat), global_gene_adj)
        
        loss_mgm_2 = loss_dam 
        
        loss_raw = reconstruction_loss(global_gene_emb.T, global_gene_edge_index, self.hidden_dim)  
        loss_aug = reconstruction_loss(global_gene_path_emb.T, global_gene_edge_index_path, self.hidden_dim)  
        loss_rec_2 = loss_raw + loss_aug 
        
        loss_cl_2 = final_cl_loss(self.a1, self.a2, global_gene_emb.T, global_gene_path_emb.T, global_gene_adj, global_gene_path_adj, tau=self.tau)
        
        
        loss_1 = self.b1 * loss_rec_1 + self.b2 * loss_mgm_1 + self.b3 * loss_cl_1    
        loss_2 = self.b1 * loss_rec_2 + self.b2 * loss_mgm_2 + self.b3 * loss_cl_2 
        
        loss = self.lambda_1 * loss_1 + self.lambda_2 * loss_2
        
        return combined_emb, loss


class ImputationRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ImputationRegressor, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, z):
        x_hat = self.fc(z)
        return x_hat


class CellTypeClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CellTypeClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, z):
        logits = self.fc(z)
        return logits

