from sklearn.model_selection import train_test_split
import torch

def get_idx(n_sample, test_valid_perc, rnd_seed=0):
    data_indices=range(n_sample)
    idx_train, idx_test = train_test_split(data_indices, test_size=test_valid_perc, random_state=rnd_seed)  #
    idx_train, idx_val = train_test_split(idx_train, test_size=len(idx_test), random_state=rnd_seed)
    return torch.LongTensor(idx_train),torch.LongTensor(idx_test),torch.LongTensor(idx_val)