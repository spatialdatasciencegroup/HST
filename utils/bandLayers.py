''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention

regression = True 
longest_window = 600
def eval_metrics(y_pred_in, y_true_in):
    eps=1e-10
    y_pred = torch.clone(y_pred_in)
    y_true = torch.clone(y_true_in)
    if regression:
        y_pred = (y_pred > 5).float()
        y_true = (y_true > 5).float()
        #print(torch.max(y_pred))
    else:
        y_pred = sigmoid(y_pred)
    #y_true = eval_label
    #y_pred = pred_prob 
    true_positive =  torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
    #false_positive = 
    gt_positive = torch.sum(torch.clamp(y_true, 0, 1))
    #gt_negative = 
    pred_positive = torch.sum(torch.round(torch.clamp(y_pred, 0, 1))) 
    
    precision = true_positive / (pred_positive + eps)
    
    recall = true_positive / (gt_positive + eps)
    
    f1_score = 2*precision*recall / (precision + recall + eps)
    
    return precision, recall, f1_score

class SelfAttention(nn.Module):
    ''' implement self attention for band seq2seq'''
    def __init__(self, embed_size, heads, device):
        super(SelfAttention, self).__init__()
        self.embed_size = 2*embed_size
        self.heads = heads
        self.head_dim = self.embed_size // heads 
        self.device = device
        
        self.vembed_size = embed_size//2
        self.vhead_dim = self.vembed_size // heads 
        
        assert (self.head_dim*self.heads == self.embed_size), 'Embed size needs to be div by heads'
        
        self.values = nn.Linear(self.vhead_dim, self.vhead_dim, bias = False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias = False)
        
        self.fc_out = nn.Linear(self.vhead_dim*self.heads, self.vembed_size)
        
    def forward(self, value, key, query, mask_ori):
        N,L = query.shape[0],query.shape[1]
        
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]
        
        value = self.values(value)
        key = self.keys(key)
        query = self.queries(query)
        
        value = value.reshape(N, value_len, self.heads, self.vhead_dim)
        key = key.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)
        
        out = []
        for i in range(self.heads):
            #indices = torch.tensor([[j for j in range(N*L) for _ in range(self.head_dim)], [j for j in range(N*L*self.head_dim)]], device = self.device)
            #sparse_query = torch.sparse_coo_tensor(indices, query[:,:,i,:].flatten(), [N*L, N*L*self.head_dim])
            query_i = query[:,:,i,:].flatten(start_dim=0, end_dim=1).unsqueeze(1) # NL *1* d_q
            #print(query_i.shape)

            key_i = torch.cat((key[:,:,i,:], torch.zeros((N, 1, self.head_dim), device = self.device)), dim = 1)# N*(L+1)*d
            #key_i = key_i.reshape()
            #print(mask.shape)
            mask = mask_ori.type(torch.int64).unsqueeze(-1)
            vmask = mask_ori.type(torch.int64).unsqueeze(-1)
            #mask = mask.unsqueeze(-1)#N*L*30
            mask = mask.expand(-1,-1,-1, self.head_dim) #N*q_L*30*d 
            s1, s2, s3, s4 = mask.shape
            mask = mask.reshape(s1,s2*s3, s4) #N*30L*d
            new_key_i = torch.gather(key_i, 1, mask).reshape(s1,s2,s3,s4).transpose(2,3).flatten(start_dim=0, end_dim=1) #, N*30L*d-> N*L*30*d ->N*L*d*30
            #new_key_i = new_key_i.reshape(s1,s2,s3,s4) #N*L*30*d

            #new_key_i = torch.transpose(new_key_i, 2,3)#N*L*d*30
            
            #new_key_i = new_key_i # NLd*30
            mask_ori = mask_ori.reshape(s1*s2,1, s3) #NL_q*1*30
            energy = torch.bmm(query_i, new_key_i)#torch.sparse.mm(sparse_query, new_key_i)
            energy = energy.masked_fill(mask_ori==longest_window, float("-1e20"))
            
            # if L == 1:
            #     energy[:,:,0] = float("-1e20")
            
            
            
            attn = torch.softmax(energy, dim=-1) #NL * 30 -> NL * 1 * 30
            #attn = attn.unsqueeze(1) #NL * 1 * 30
            
            value_i = torch.cat((value[:,:,i,:], torch.zeros((N, 1, self.vhead_dim), device = self.device)), dim = 1)# N*(L+1)*d
            
            
            #mask = mask.unsqueeze(-1)#N*L*30
            vmask = vmask.expand(-1,-1,-1, self.vhead_dim) #N*q_L*30*d 
            v1, v2, v3, v4 = vmask.shape
            vmask = vmask.reshape(v1,v2*v3, v4) #N*30L*d
            
            new_value_i = torch.gather(value_i, 1, vmask).reshape(v1,v2,v3,v4).flatten(start_dim=0, end_dim=1) # N*30L*d
            #new_value_i = new_value_i.reshape(s1,s2,s3,s4) #N*L*30*d
            #new_value_i = new_value_i # NL*30*d
            
            out_i = torch.bmm(attn, new_value_i).reshape(N, L, self.vhead_dim) #NL*d
            #out_i = out_i.reshape(N, L, self.head_dim)
            
            if i==0:
                out = out_i
            else:
                out = torch.cat((out, out_i),dim = -1)
#             query_i = query[:,:,i,:]
#             key_i = key[:,:,i,:]
#             value_i = value[:,:,i,:]
#         # index_row = torch.tensor([[i for j in range(self.head_dim)] for i in range(N*L)]).flatten()
#         # index_col = torch.tensor([i for i in range(N*L*self.head_dim)])
#         # indices = torch.tensor([index_row, index_col])        
#             index_row = [i for i in range(N*L) for _ in range(self.head_dim)]
#             index_col = [i for i in range(N*L*self.head_dim)]
#             indices = torch.tensor([index_row, index_col], device = self.device)
#             sparse_query = torch.sparse_coo_tensor(indices, query_i.flatten(), [N*L, N*L*self.head_dim])

#             key_i = torch.cat((key_i, torch.zeros((N, 1, self.head_dim), device = self.device)), dim = 1)# N*(L+1)*d
#             #key_i = key_i.reshape()
#             mask = mask.type(torch.int64)
#             mask = mask.unsqueeze(-1)#N*L*30
#             mask = mask.expand(-1,-1,-1, self.head_dim) #N*30L*d 
#             s1, s2, s3, s4 = mask.shape
#             mask = mask.reshape(s1,s2*s3, s4) #N*30L*d
#             new_key_i = torch.gather(key_i, 1, mask) # N*30L*d
#             new_key_i = new_key_i.reshape(s1,s2,s3,s4) #N*L*30*d

#             new_key_i = torch.transpose(new_key_i, 2,3)#N*L*d*30
            
#             new_key_i = new_key_i.flatten(start_dim=0, end_dim=2) # NLd*30
            
#             attn = torch.softmax(torch.sparse.mm(sparse_query, new_key_i), dim=-1) #NL * 30
#             attn = attn.unsqueeze(1) #NL * 1 * 30
            
#             value_i = torch.cat((value_i, torch.zeros((N, 1, self.head_dim), device = self.device)), dim = 1)# N*(L+1)*d
            
#             new_value_i = torch.gather(value_i, 1, mask) # N*30L*d
#             new_value_i = new_value_i.reshape(s1,s2,s3,s4) #N*L*30*d
#             new_value_i = new_value_i.flatten(start_dim=0, end_dim=1) # NL*30*d
            
#             out_i = torch.bmm(attn, new_value_i) #NL*d
#             out_i = out_i.reshape(N, L, self.head_dim)
            
#             if i==0:
#                 out = out_i
#             else:
#                 out = torch.cat((out, out_i),dim = -1)
            
            
            
        
            
            
        
#         energy = torch.einsum("nqhd, nkhd->nhqk", [query, key])
        
#         #query shape: (N, query_len, heads, head_dim)
#         #key shape: (N, key_len, heads, head_dim)
#         #value shape: (N, val_len, heads, head_dim)
#         mask = mask.unsqueeze(1)
#         if mask is not None:
#             energy = energy.masked_fill(mask, float("-1e20"))
#         attention = torch.softmax(energy/0.1, dim = 3)
        
#         out = torch.einsum("nhql, nlhd->nqhd",[attention, value])

#         out = out.reshape(N, query_len, self.heads*self.head_dim)
        
#         #attention shape: (N, heads, query_len, key_len)
#         #value shape: (N, val_len, heads, head_dim)
#         #after einsum: (N, query_len, heads, head_dim)
        
        out = self.fc_out(out)
        
        return out, attn.squeeze().reshape(N,L, -1)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, device):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads, device)
        
        self.vembed_size = int(embed_size/2)
        self.norm1 = nn.LayerNorm(self.vembed_size)
        self.norm2 = nn.LayerNorm(self.vembed_size)
        
        
        
        self.linear2 = nn.Linear(self.vembed_size, self.vembed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(2*embed_size, forward_expansion*2*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*2*embed_size,embed_size),
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, value, key, query, mask):
        
        attention, weights =  self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query[:,:,:self.vembed_size]))
        #feed_forward = self.feed_forward(x)
        # out = self.dropout(self.norm2(feed_forward + x))
        #out = self.dropout(self.norm2(self.linear2 (x)))
        
        return x, weights
    
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length, gamma=2):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        print(src_vocab_size, embed_size)
        self.band_embedding = nn.Linear(src_vocab_size, embed_size)
        
        
        self.gamma = gamma 
        self.time_position_encoding = TimeEncoder(expand_dim = embed_size)#TimeEncode(expand_dim= int(self.space_time_dim/2)) ##to do
        self.space_position_encoding = SpaceEncoder( G=1, M=2, F_dim=embed_size, H_dim=2*embed_size,
                                          D = embed_size, gamma =self.gamma)        #SpaceEncoder( G=1, M=2, F_dim=self.space_dim, H_dim=2*self.space_dim,
                                          #D = self.space_time_dim - int(self.space_time_dim / 2), gamma =self.gamma)        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout = dropout, forward_expansion = forward_expansion, device = device)
            ]
        
        
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask, downsample, time_pos, space_pos):
        N, seq_len,_ = x.shape
        time_positions = self.time_position_encoding(time_pos)
        #space_positions = space_pos#self.space_position_encoding(space_pos)
        x = torch.cat((x ,space_pos), dim= -1)
        
        x = x.reshape(N*seq_len, -1)
        
        x_new = torch.sparse.mm(downsample, x)
        
        #x = torch.einsum('bij,bjk->bik',downsample, x)
        x_new = x_new.reshape(N, longest_window, -1) ### x_new contain more sequence point, internal point + leaf node
        
        
        x = x.reshape(N, seq_len, -1) ### x only contains leaf node query points, 
        out = x# self.dropout(x)
        
        for layer in self.layers:
            out, _ = layer( x_new[:,:,:int(self.embed_size/2)], x_new,out, mask)
            
            out = torch.cat((out ,space_pos), dim= -1)
            out = out.reshape(N*seq_len, -1)
            x_new = torch.sparse.mm(downsample, out)
            x_new = x_new.reshape(N, longest_window, -1)
            out = out.reshape(N, seq_len, -1)
            
            #out = torch.einsum('bij,bjk->bik',downsample, )
        return x_new
    
        
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads, device = device)
        self.norm = nn.LayerNorm(2*embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout,forward_expansion, device = device)
        self.dropout = nn.Dropout(dropout)

        
    def forward(self, query, value, key, trg_mask):
        #attention = self.attention(x,x,x,trg_mask)
        #query = x #self.dropout(self.norm(attention + x))
        out, weights = self.transformer_block(value, key, query, trg_mask)
        return out, weights

class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, 
                num_layers, heads,
                forward_expansion,
                dropout,
                device,
                max_length,
                gamma = 2):
        super(Decoder, self).__init__()
        self.device = device
        self.gamma = gamma 
        self.embed_size = embed_size
        self.band_embedding = nn.Linear(trg_vocab_size, embed_size)
        #self.position_embedding = TimeEncoder(expand_dim = embed_size) # TO do
        self.time_position_encoding = TimeEncoder(expand_dim = embed_size)#TimeEncode(expand_dim= int(self.space_time_dim/2)) ##to do
        self.space_position_encoding = SpaceEncoder( G=1, M=2, F_dim=embed_size, H_dim=2*embed_size,
                                          D = embed_size, gamma =self.gamma)        #SpaceEncoder( G=1, M=2, F_dim=self.space_dim, H_dim=2*self.space_dim,
                                          #D = self.space_time_dim - int(self.space_time_dim / 2), gamma =self.gamma)        
        
        self.layers = nn.ModuleList(
        [
            DecoderBlock(embed_size, heads, forward_expansion, dropout, device) for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(int(embed_size/2), 1)
        self.dropout = nn.Dropout(dropout)
        
    #def forward(self, x, enc_out, src_mask, trg_mask,position):
    def forward(self, x,trg_time_pos, space_positions, enc_out, trg_mask, enc_space_positions):
        N, seq_length,_ = x.shape
        time_positions = self.time_position_encoding(trg_time_pos)
        #space_positions = trg_space_pos #self.space_position_encoding(trg_space_pos)
#         positions = self.position_embedding(position)
        #x = x#self.dropout(self.band_embedding(x))
    
        #enc_space_positions = src_space_pos #self.space_position_encoding(src_space_pos)
        #enc_out = torch.cat((enc_out, enc_space_positions), dim = -1)
        
        for layer in self.layers:
            x = torch.cat((x, space_positions), dim = -1)
            x, weights = layer(x, enc_out[:,:,:int(self.embed_size/2)], enc_out, trg_mask)
            
        out = self.fc_out(x)
        
        return out, weights
    
class Transformer(nn.Module):
    def __init__(self,
                src_vocab_size,
                trg_vocab_size,
                src_pad_idx,
                trg_pad_idx,
                embed_size = 256,
                num_layers = 2,
                forward_expansion =2,
                heads = 2,
                dropout = 0,
                device = "cuda",
                max_length = 10,
                gamma=0.05):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )
        
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )
        
        self.space_position_encoding = SpaceEncoder( G=1, M=2, F_dim=int(embed_size*3/2), H_dim=2*embed_size,
                                          D = embed_size, gamma = gamma)        #SpaceEncoder( G=1, M=2, F_dim=self.space_dim, H_dim=2*self.space_dim,
         
        self.band_embedding = nn.Linear(trg_vocab_size, int(embed_size/2))
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.dropout = nn.Dropout(dropout)
        
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        #print()
        s0, s1, s2, s3, s4 = src_mask.shape
        src_mask = src_mask.reshape((s0,s1,s2,s3))
#         print()
        # (N, 1, 1, src_len)
        
        return src_mask.to(self.device)
        
    def make_trg_mask(self, trg):
        N, trg_len,_ = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        
        return trg_mask.to(self.device)
    
    def forward(self, src, trg, downsample, src_time_pos, src_space_pos,trg_time_pos, trg_space_pos,src_mask,trg_mask):
        #print('start')
        #src_mask = self.make_src_mask(src)
        #trg_mask = self.make_trg_mask(trg)
        src = self.band_embedding(src)
        trg = self.band_embedding(trg)
        src_space_position_encodings = self.space_position_encoding(src_space_pos)
        trg_space_position_encodings = self.space_position_encoding(trg_space_pos)
        
        enc_src = self.encoder(src, src_mask, downsample, src_time_pos, src_space_position_encodings)
        out, weights = self.decoder(trg,trg_time_pos, trg_space_position_encodings, enc_src, trg_mask, src_space_position_encodings)
        
        return out, weights
    
                 
class TimeEncoder(nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncoder, self).__init__()
        #init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])
        
        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 **(np.linspace(1, 4, time_dim) ))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())
        
        #self.dense = torch.nn.Linear(time_dim, expand_dim, bias=False)

        #torch.nn.init.xavier_normal_(self.dense.weight)
        
    def forward(self, ts):
        # ts: [N, L]
        
        #print("original time series: ", ts)
        batch_size = ts.size(0)
        seq_len = ts.size(1)
                
        ts = ts.view(batch_size, seq_len, 1)# [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1) # [N, L, time_dim]
        #print("after basis freq: ", map_ts)
        map_ts += self.phase.view(1, 1, -1)
        #print("after phase: ", map_ts)
        harmonic = torch.cos(map_ts)
        
        return harmonic        

class SpaceEncoder(nn.Module):
    def __init__(self, G: int, M: int, F_dim: int, H_dim: int, D: int, gamma: float):
        """
        Learnable Fourier Features from https://arxiv.org/pdf/2106.02795.pdf (Algorithm 1)
        Implementation of Algorithm 1: Compute the Fourier feature positional encoding of a multi-dimensional position
        Computes the positional encoding of a tensor of shape [N, G, M]
        :param G: positional groups (positions in different groups are independent)
        :param M: each point has a M-dimensional positional values
        :param F_dim: depth of the Fourier feature dimension
        :param H_dim: hidden layer dimension
        :param D: positional encoding dimension
        :param gamma: parameter to initialize Wr
        """
        super().__init__()
        self.G = G  # 1
        self.M = M  #2
        self.F_dim = F_dim # space dim
        self.H_dim = H_dim 
        self.D = D
        self.gamma = gamma

        # Projection matrix on learned lines (used in eq. 2)
        self.Wr = nn.Linear(self.M, self.F_dim // 2, bias=False)
        # MLP (GeLU(F @ W1 + B1) @ W2 + B2 (eq. 6)
        self.mlp = nn.Sequential(
            nn.Linear(self.F_dim, self.H_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.H_dim, self.D // self.G)
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)

    def forward(self, x):
        """
        Produce positional encodings from x
        :param x: tensor of shape [N, G, M] that represents N positions where each position is in the shape of [G, M],
                  where G is the positional group and each group has M-dimensional positional values.
                  Positions in different positional groups are independent
        :return: positional encoding for X
        """
        N, G, M = x.shape
        # Step 1. Compute Fourier features (eq. 2)
        projected = self.Wr(x)
        cosines = torch.cos(projected)
        sines = torch.sin(projected)
        F = 1 / np.sqrt(self.F_dim) * torch.cat([cosines, sines], dim=-1)
        # Step 2. Compute projected Fourier features (eq. 6)
    
        #Y = self.mlp(F)
        
        # Step 3. Reshape to x's shape
        #PEx = Y.reshape((N, self.D))
        PEx = F.reshape((N, G, self.F_dim))
        return PEx
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
