import os
import matplotlib
import numpy as np
import pandas as pd
import torch
import utils as utils

from scipy import special
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class USHCN(object):
    """
    variables:
    "SNOW","SNWD","PRCP","TMAX","TMIN"
    """
    def __init__(self, root, n_samples = None, device = torch.device("cpu")):

        self.root = root
        self.device = device

        self.process()

        if device == torch.device("cpu"):
            self.data = torch.load(os.path.join(self.processed_folder, 'ushcn.pt'), map_location='cpu')
        else:
            self.data = torch.load(os.path.join(self.processed_folder, 'ushcn.pt'))

        if n_samples is not None:
            print('Total records:', len(self.data))
            self.data = self.data[:n_samples]

    def process(self):
        if self._check_exists():
            return
        
        filename = os.path.join(self.raw_folder, 'small_chunked_sporadic.csv')
        
        os.makedirs(self.processed_folder, exist_ok=True)

        print('Processing {}...'.format(filename))

        full_data = pd.read_csv(filename, index_col=0)
        full_data.index = full_data.index.astype('int32')

        entities = []
        value_cols = [c.startswith('Value') for c in full_data.columns]
        value_cols = list(full_data.columns[value_cols])
        mask_cols = [('Mask' + x[5:]) for x in value_cols]
        # print(value_cols)
        # print(mask_cols)
        data_gp = full_data.groupby(level=0) # group by index
        for record_id, data in data_gp:
            tt = torch.tensor(data['Time'].values).to(self.device).float() * (48./200)
            sorted_inds = tt.argsort() # sort over time
            vals = torch.tensor(data[value_cols].values).to(self.device).float()
            mask = torch.tensor(data[mask_cols].values).to(self.device).float()
            entities.append((record_id, tt[sorted_inds], vals[sorted_inds], mask[sorted_inds]))

        torch.save(
            entities,
            os.path.join(self.processed_folder, 'ushcn.pt')
        )

        print('Total records:', len(entities))

        print('Done!')

    def _check_exists(self):

        if not os.path.exists(os.path.join(self.processed_folder, 'ushcn.pt')):
            return False
        
        return True

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')
        
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    


def USHCN_time_chunk(data, args, device):

    chunk_data = []

    for b, (record_id, tt, vals, mask) in enumerate(data):
        for st in range(0, args.n_months - args.history - args.pred_window + 1, args.pred_window):
            et = st + args.history + args.pred_window
            if(et == args.n_months):
                indices = torch.where((tt >= st) & (tt <= et))[0]
            else:
                indices = torch.where((tt >= st) & (tt < et))[0]

            t_bias = torch.tensor(st).to(device)
            chunk_data.append((record_id, tt[indices]-t_bias, vals[indices], mask[indices], t_bias))

    return chunk_data


def USHCN_patch_variable_time_collate_fn(batch, args, device = torch.device("cpu"), data_type = "train", 
    data_min = None, data_max = None, time_max = None):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask) where
        - record_id is a patient id
        - tt is a (T, ) tensor containing T time values of observations.
        - vals is a (T, D) tensor containing observed values for D variables.
        - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
    Returns:
    Data form as input:
        batch_tt: (B, M, L_in, D) the batch contains a maximal L_in time values of observations among M patches.
        batch_vals: (B, M, L_in, D) tensor containing the observed values.
        batch_mask: (B, M, L_in, D) tensor containing 1 where values were observed and 0 otherwise.
    Data form to predict:
        flat_tt: (L_out) the batch contains a maximal L_out time values of observations.
        flat_vals: (B, L_out, D) tensor containing the observed values.
        flat_mask: (B, L_out, D) tensor containing 1 where values were observed and 0 otherwise.
    """

    D = batch[0][2].shape[1]
    # combined_tt shape is (T_o, )
    combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
    # print(combined_tt.max(), combined_tt.min())
    # print(inverse_indices.shape, np.sum([len(ex[1]) for ex in batch]), inverse_indices.max())
    # print(inverse_indices)

    # the number of observed time points 
    n_observed_tp = torch.lt(combined_tt, args.history).sum()
    observed_tp = combined_tt[:n_observed_tp] # (n_observed_tp, )
    # print(n_observed_tp, len(combined_tt)-n_observed_tp)
    # print(combined_tt[:n_observed_tp])
    # print(combined_tt[n_observed_tp:])

    patch_indices = []
    st, ed = 0, args.patch_size
    for i in range(args.npatch):
        if(i == args.npatch-1):
            inds = torch.where((observed_tp >= st) & (observed_tp <= ed))[0]
        else:
            inds = torch.where((observed_tp >= st) & (observed_tp < ed))[0]
        patch_indices.append(inds)
        # print(st, ed, observed_tp[inds[0]: inds[-1]+1])

        st += args.stride
        ed += args.stride

    offset = 0
    combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    predicted_tp = []
    predicted_data = []
    predicted_mask = [] 
    batch_t_bias = []
    for b, (record_id, tt, vals, mask, t_bias) in enumerate(batch):
        batch_t_bias.append(t_bias)

        indices = inverse_indices[offset:offset+len(tt)]
        offset += len(tt)
        combined_vals[b, indices] = vals
        combined_mask[b, indices] = mask

        tmp_n_observed_tp = torch.lt(tt, args.history).sum()
        predicted_tp.append(tt[tmp_n_observed_tp:])
        predicted_data.append(vals[tmp_n_observed_tp:])
        predicted_mask.append(mask[tmp_n_observed_tp:])

    combined_tt = combined_tt[:n_observed_tp] # (T_o, )
    combined_vals = combined_vals[:, :n_observed_tp]
    combined_mask = combined_mask[:, :n_observed_tp]
    predicted_tp = pad_sequence(predicted_tp, batch_first=True)
    predicted_data = pad_sequence(predicted_data, batch_first=True)
    predicted_mask = pad_sequence(predicted_mask, batch_first=True)


    combined_tt = utils.normalize_masked_tp(combined_tt, att_min = 0, att_max = time_max)
    predicted_tp = utils.normalize_masked_tp(predicted_tp, att_min = 0, att_max = time_max)
    # print(predicted_data.sum(), predicted_tp.sum())
    batch_t_bias = torch.stack(batch_t_bias) # (n_batch, )
    batch_t_bias = utils.normalize_masked_tp(batch_t_bias, att_min = 0, att_max = time_max)
        
    data_dict = {
        "data": combined_vals, # (n_batch, T_o, D)
        "time_steps": combined_tt, # (T_o, )
        "mask": combined_mask, # (n_batch, T_o, D)
        "data_to_predict": predicted_data, # (n_batch, T, D)
        "tp_to_predict": predicted_tp, # (B, T)
        "mask_predicted_data": predicted_mask, # (n_batch, T, D)
        }

    data_dict = utils.split_and_patch_batch(data_dict, args, n_observed_tp, patch_indices)
    # print("patchdata:", data_dict["data_to_predict"].sum(), data_dict["mask_predicted_data"].sum())

    # print(batch_t_bias.shape, data_dict["observed_tp"].shape, data_dict["tp_to_predict"].shape)
    data_dict["observed_tp"] = data_dict["observed_tp"] + batch_t_bias.view(len(batch_t_bias), 1, 1, 1)
    # data_dict["observed_tp"] = data_dict["observed_tp"] * (data_dict["mask_predicted_data"].sum(dim=-1)>1e-8)

    data_dict["tp_to_predict"] = data_dict["tp_to_predict"] + batch_t_bias.view(len(batch_t_bias), 1)
    data_dict["tp_to_predict"][data_dict["mask_predicted_data"].sum(dim=-1)<1e-8] = 0
    # delta = data_dict["tp_to_predict"].view(len(batch_t_bias),-1).max(dim=-1)[0] - data_dict["observed_tp"].view(len(batch_t_bias),-1).min(dim=-1)[0]
    # delta = data_dict["tp_to_predict"].view(len(batch_t_bias),-1).min(dim=-1)[0] - data_dict["observed_tp"].view(len(batch_t_bias),-1).max(dim=-1)[0]
    # print((delta*48).max(), (delta*48).min())

    return data_dict


def USHCN_patch_variable_time_collate_fn_max(batch, args, device = torch.device("cpu"), data_type = "train", 
    data_min = None, data_max = None, time_max = None):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask) where
        - record_id is a patient id
        - tt is a (T, ) tensor containing T time values of observations.
        - vals is a (T, D) tensor containing observed values for D variables.
        - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
    Returns:
    Data form as input:
        batch_tt: (B, M, L_in, D) the batch contains a maximal L_in time values of observations among M patches.
        batch_vals: (B, M, L_in, D) tensor containing the observed values.
        batch_mask: (B, M, L_in, D) tensor containing 1 where values were observed and 0 otherwise.
    Data form to predict:
        flat_tt: (L_out) the batch contains a maximal L_out time values of observations.
        flat_vals: (B, L_out, D) tensor containing the observed values.
        flat_mask: (B, L_out, D) tensor containing 1 where values were observed and 0 otherwise.
    """

    D = batch[0][2].shape[1]
    # combined_tt shape is (T_o, )
    combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
    # print(combined_tt.max(), combined_tt.min())
    # print(inverse_indices.shape, np.sum([len(ex[1]) for ex in batch]), inverse_indices.max())
    # print(inverse_indices)

    # the number of observed time points 
    n_observed_tp = torch.lt(combined_tt, args.history).sum()
    observed_tp = combined_tt[:n_observed_tp] # (n_observed_tp, )
    # print(n_observed_tp, len(combined_tt)-n_observed_tp)
    # print(combined_tt[:n_observed_tp])
    # print(combined_tt[n_observed_tp:])

    patch_indices = []
    st, ed = 0, args.patch_size
    for i in range(args.npatch):
        if(i == args.npatch-1):
            inds = torch.where((observed_tp >= st) & (observed_tp <= ed))[0]
        else:
            inds = torch.where((observed_tp >= st) & (observed_tp < ed))[0]
        patch_indices.append(inds)
        # print(st, ed, observed_tp[inds[0]: inds[-1]+1])

        st += args.stride
        ed += args.stride

    offset = 0
    combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    predicted_tp = []
    predicted_data = []
    predicted_mask = [] 
    batch_t_bias = []
    for b, (record_id, tt, vals, mask, t_bias) in enumerate(batch):
        batch_t_bias.append(t_bias)

        indices = inverse_indices[offset:offset+len(tt)]
        offset += len(tt)
        combined_vals[b, indices] = vals
        combined_mask[b, indices] = mask

        tmp_n_observed_tp = torch.lt(tt, args.history).sum()
        predicted_tp.append(tt[tmp_n_observed_tp:])
        predicted_data.append(vals[tmp_n_observed_tp:])
        predicted_mask.append(mask[tmp_n_observed_tp:])

    combined_tt = combined_tt[:n_observed_tp] # (T_o, )
    combined_vals = combined_vals[:, :n_observed_tp]
    combined_mask = combined_mask[:, :n_observed_tp]
    predicted_tp = pad_sequence(predicted_tp, batch_first=True)
    predicted_data = pad_sequence(predicted_data, batch_first=True)
    predicted_mask = pad_sequence(predicted_mask, batch_first=True)

    # (핵심 변경) parse_datasets에서 미리 계산한 전역 최대 패치 길이를 사용
    patch_maxlen = args.maxlen
    
    final_patch_data, final_patch_mask, final_patch_tp = [], [], []

    # 각 패치 인덱스에 대해 루프를 돌며 데이터를 슬라이싱하고 `patch_maxlen`으로 패딩
    for inds in patch_indices:
        current_patch_len = len(inds)

        # 현재 패치에 해당하는 데이터를 잘라냄
        patch_vals_unpadded = combined_vals[:, inds, :]
        patch_mask_unpadded = combined_mask[:, inds, :]
        patch_tp_unpadded = combined_tt[inds]

        # `patch_maxlen` 길이를 갖는 0으로 채워진 템플릿 텐서 생성
        padded_vals = torch.zeros(B, patch_maxlen, D, device=device)
        padded_mask = torch.zeros(B, patch_maxlen, D, device=device)
        padded_tp = torch.zeros(patch_maxlen, device=device)

        # 템플릿에 실제 데이터 복사 (patch_maxlen보다 길면 잘림)
        copy_len = min(current_patch_len, patch_maxlen)
        padded_vals[:, :copy_len, :] = patch_vals_unpadded[:, :copy_len, :]
        padded_mask[:, :copy_len, :] = patch_mask_unpadded[:, :copy_len, :]
        padded_tp[:copy_len] = patch_tp_unpadded[:copy_len]

        # 패딩 완료된 텐서를 리스트에 추가
        final_patch_data.append(padded_vals)
        final_patch_mask.append(padded_mask)
        final_patch_tp.append(padded_tp)
    
    # 패딩된 패치들을 하나의 텐서로 결합
    if final_patch_data:
        # 리스트에 담긴 각 패치 텐서들을 새로운 차원(dim=1)으로 쌓음
        final_patch_data = torch.stack(final_patch_data, dim=1)  # (B, M, L, D)
        final_patch_mask = torch.stack(final_patch_mask, dim=1)  # (B, M, L, D)
        final_patch_tp = torch.stack(final_patch_tp, dim=0)      # (M, L)
    else: # 패치가 없는 경우 빈 텐서 생성
        M = args.npatch
        final_patch_data = torch.zeros(B, M, patch_maxlen, D, device=device)
        final_patch_mask = torch.zeros(B, M, patch_maxlen, D, device=device)
        final_patch_tp = torch.zeros(M, patch_maxlen, device=device)

    combined_tt = utils.normalize_masked_tp(final_patch_tp, att_min = 0, att_max = time_max)
    predicted_tp = utils.normalize_masked_tp(predicted_tp, att_min = 0, att_max = time_max)
    # print(predicted_data.sum(), predicted_tp.sum())
    batch_t_bias = torch.stack(batch_t_bias) # (n_batch, )
    batch_t_bias = utils.normalize_masked_tp(batch_t_bias, att_min = 0, att_max = time_max)
        
    data_dict = {
        "data": combined_vals, # (n_batch, T_o, D)
        "time_steps": combined_tt, # (T_o, )
        "mask": combined_mask, # (n_batch, T_o, D)
        "data_to_predict": predicted_data, # (n_batch, T, D)
        "tp_to_predict": predicted_tp, # (B, T)
        "mask_predicted_data": predicted_mask, # (n_batch, T, D)
        }

    data_dict = utils.split_and_patch_batch(data_dict, args, n_observed_tp, patch_indices)
    # print("patchdata:", data_dict["data_to_predict"].sum(), data_dict["mask_predicted_data"].sum())

    # print(batch_t_bias.shape, data_dict["observed_tp"].shape, data_dict["tp_to_predict"].shape)
    data_dict["observed_tp"] = data_dict["observed_tp"] + batch_t_bias.view(len(batch_t_bias), 1, 1, 1)
    # data_dict["observed_tp"] = data_dict["observed_tp"] * (data_dict["mask_predicted_data"].sum(dim=-1)>1e-8)

    data_dict["tp_to_predict"] = data_dict["tp_to_predict"] + batch_t_bias.view(len(batch_t_bias), 1)
    data_dict["tp_to_predict"][data_dict["mask_predicted_data"].sum(dim=-1)<1e-8] = 0
    # delta = data_dict["tp_to_predict"].view(len(batch_t_bias),-1).max(dim=-1)[0] - data_dict["observed_tp"].view(len(batch_t_bias),-1).min(dim=-1)[0]
    # delta = data_dict["tp_to_predict"].view(len(batch_t_bias),-1).min(dim=-1)[0] - data_dict["observed_tp"].view(len(batch_t_bias),-1).max(dim=-1)[0]
    # print((delta*48).max(), (delta*48).min())

    return data_dict


def USHCN_variable_time_collate_fn(batch, args, device = torch.device("cpu"), data_type = "train", 
    data_min = None, data_max = None, time_max = None):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask) where
        - record_id is a patient id
        - tt is a (T, ) tensor containing T time values of observations.
        - vals is a (T, D) tensor containing observed values for D variables.
        - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
    Returns:
        batch_tt: (B, L) the batch contains a maximal L time values of observations.
        batch_vals: (B, L, D) tensor containing the observed values.
        batch_mask: (B, L, D) tensor containing 1 where values were observed and 0 otherwise.
    """

    # n_observed_tps = []
    observed_tp = []
    observed_data = []
    observed_mask = [] 
    predicted_tp = []
    predicted_data = []
    predicted_mask = [] 
    # batch_t_bias = []

    for b, (record_id, tt, vals, mask, t_bias) in enumerate(batch):
        # batch_t_bias.append(t_bias)
        n_observed_tp = torch.lt(tt, args.history).sum()
        # print(len(tt), n_observed_tp)
        # n_observed_tps.append(n_observed_tp)
        tt = tt + t_bias
        observed_tp.append(tt[:n_observed_tp])
        observed_data.append(vals[:n_observed_tp])
        observed_mask.append(mask[:n_observed_tp])
        
        predicted_tp.append(tt[n_observed_tp:])
        predicted_data.append(vals[n_observed_tp:])
        predicted_mask.append(mask[n_observed_tp:])

    observed_tp = pad_sequence(observed_tp, batch_first=True)
    observed_data = pad_sequence(observed_data, batch_first=True)
    observed_mask = pad_sequence(observed_mask, batch_first=True)
    predicted_tp = pad_sequence(predicted_tp, batch_first=True)
    predicted_data = pad_sequence(predicted_data, batch_first=True)
    predicted_mask = pad_sequence(predicted_mask, batch_first=True)
    # print(observed_tp.shape, observed_data.shape, observed_mask.shape,\
    #     predicted_tp.shape, predicted_data.shape, predicted_mask.shape)
    
    observed_tp = utils.normalize_masked_tp(observed_tp, att_min = 0, att_max = time_max)
    predicted_tp = utils.normalize_masked_tp(predicted_tp, att_min = 0, att_max = time_max)
    # print(predicted_data.sum(), predicted_tp.sum())
    # batch_t_bias = torch.stack(batch_t_bias) # (n_batch, )
    # batch_t_bias = utils.normalize_masked_tp(batch_t_bias, att_min = 0, att_max = time_max)

    # print(observed_tp.max())
    # print(predicted_tp.max())

    # print(batch_t_bias.shape, observed_tp.shape, predicted_tp.shape)
    # observed_tp = observed_tp + batch_t_bias.view(len(batch_t_bias), 1)
    # observed_tp[observed_mask.sum(dim=-1)<1e-8] = 0
    # predicted_tp = predicted_tp + batch_t_bias.view(len(batch_t_bias), 1)
    # predicted_tp[predicted_mask.sum(dim=-1)<1e-8] = 0
        
    data_dict = {"observed_data": observed_data,
            "observed_tp": observed_tp,
            "observed_mask": observed_mask,
            "data_to_predict": predicted_data,
            "tp_to_predict": predicted_tp,
            "mask_predicted_data": predicted_mask,
            }
    # print("vecdata:", data_dict["data_to_predict"].sum(), data_dict["mask_predicted_data"].sum())
    
    return data_dict


def USHCN_variable_time_collate_fn_max(batch, args, device = torch.device("cpu"), data_type = "train", 
    data_min = None, data_max = None, time_max = None):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask) where
        - record_id is a patient id
        - tt is a (T, ) tensor containing T time values of observations.
        - vals is a (T, D) tensor containing observed values for D variables.
        - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
    Returns:
        batch_tt: (B, L) the batch contains a maximal L time values of observations.
        batch_vals: (B, L, D) tensor containing the observed values.
        batch_mask: (B, L, D) tensor containing 1 where values were observed and 0 otherwise.
    """

    # n_observed_tps = []
    observed_tp = []
    observed_data = []
    observed_mask = [] 
    predicted_tp = []
    predicted_data = []
    predicted_mask = [] 
    # batch_t_bias = []

    for b, (record_id, tt, vals, mask, t_bias) in enumerate(batch):
        # batch_t_bias.append(t_bias)
        n_observed_tp = torch.lt(tt, args.history).sum()
        # print(len(tt), n_observed_tp)
        # n_observed_tps.append(n_observed_tp)
        tt = tt + t_bias
        observed_tp.append(tt[:n_observed_tp])
        observed_data.append(vals[:n_observed_tp])
        observed_mask.append(mask[:n_observed_tp])
        
        predicted_tp.append(tt[n_observed_tp:])
        predicted_data.append(vals[n_observed_tp:])
        predicted_mask.append(mask[n_observed_tp:])

    # ==================================================================
    # ### 핵심 수정 부분: observed 데이터를 args.ts_len으로 수동 패딩 ###
    # ==================================================================
    
    # 1. 패딩에 필요한 정보 가져오기
    batch_size = len(batch)
    num_features = batch[0][2].shape[1] # vals의 feature dimension (D)
    max_len = args.maxlen

    # 2. 목표 길이(max_len)를 갖는 0으로 채워진 템플릿 텐서 생성
    padded_observed_data = torch.zeros(batch_size, max_len, num_features, device=device)
    padded_observed_mask = torch.zeros(batch_size, max_len, num_features, device=device)
    padded_observed_tp = torch.zeros(batch_size, max_len, device=device)

    # 3. 루프를 돌며 각 시퀀스의 실제 데이터를 템플릿 텐서에 복사
    for i in range(batch_size):
        # 현재 데이터의 실제 길이 (max_len보다 길 경우 잘라냄)
        length = min(observed_data[i].shape[0], max_len)
        
        padded_observed_data[i, :length, :] = observed_data[i][:length]
        padded_observed_mask[i, :length, :] = observed_mask[i][:length]
        padded_observed_tp[i, :length] = observed_tp[i][:length]

    # 4. 패딩 완료된 텐서를 원래 변수 이름으로 할당
    observed_data = padded_observed_data
    observed_mask = padded_observed_mask
    observed_tp = padded_observed_tp

    # ==================================================================
    # ### 수정 끝 ###
    # ==================================================================
    
    predicted_tp = pad_sequence(predicted_tp, batch_first=True)
    predicted_data = pad_sequence(predicted_data, batch_first=True)
    predicted_mask = pad_sequence(predicted_mask, batch_first=True)
    # print(observed_tp.shape, observed_data.shape, observed_mask.shape,\
    #     predicted_tp.shape, predicted_data.shape, predicted_mask.shape)
    
    observed_tp = utils.normalize_masked_tp(observed_tp, att_min = 0, att_max = time_max)
    predicted_tp = utils.normalize_masked_tp(predicted_tp, att_min = 0, att_max = time_max)
    # print(predicted_data.sum(), predicted_tp.sum())
    # batch_t_bias = torch.stack(batch_t_bias) # (n_batch, )
    # batch_t_bias = utils.normalize_masked_tp(batch_t_bias, att_min = 0, att_max = time_max)

    # print(observed_tp.max())
    # print(predicted_tp.max())

    # print(batch_t_bias.shape, observed_tp.shape, predicted_tp.shape)
    # observed_tp = observed_tp + batch_t_bias.view(len(batch_t_bias), 1)
    # observed_tp[observed_mask.sum(dim=-1)<1e-8] = 0
    # predicted_tp = predicted_tp + batch_t_bias.view(len(batch_t_bias), 1)
    # predicted_tp[predicted_mask.sum(dim=-1)<1e-8] = 0
        
    data_dict = {"observed_data": observed_data,
            "observed_tp": observed_tp,
            "observed_mask": observed_mask,
            "data_to_predict": predicted_data,
            "tp_to_predict": predicted_tp,
            "mask_predicted_data": predicted_mask,
            }
    # print("vecdata:", data_dict["data_to_predict"].sum(), data_dict["mask_predicted_data"].sum())
    
    return data_dict


def USHCN_get_seq_length(args, records):
    
    max_input_len = 0
    max_pred_len = 0
    lens = []
    for b, (record_id, tt, vals, mask, t_bias) in enumerate(records):
        n_observed_tp = torch.lt(tt, args.history).sum()
        max_input_len = max(max_input_len, n_observed_tp)
        max_pred_len = max(max_pred_len, len(tt) - n_observed_tp)
        lens.append(n_observed_tp)
    lens = torch.stack(lens, dim=0)
    median_len = lens.median()

    return max_input_len, max_pred_len, median_len



