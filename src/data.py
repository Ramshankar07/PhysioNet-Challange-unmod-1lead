# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)

import numpy as np
from scipy.io import loadmat
import os
import torch

from src.preprocess import preprocess_signal, preprocess_label
from src.evaluate import load_weights


class ecg_dataset(torch.utils.data.Dataset):
    """ Pytorch dataloader for ecg training and evaluation """
    def __init__(self, filenames, sample_rates, Y, preprocess_cfg):
        self.filenames = filenames
        self.sample_rates = sample_rates
        self.Y = Y
        self.preprocess_cfg = preprocess_cfg

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        # Load and preprocess the full ECG data
        full_ecg = preprocess_signal(load_data(self.filenames[i]), self.preprocess_cfg, self.sample_rates[i])
        
        # Ensure we have at least one channel, if not, add a channel dimension
        if full_ecg.ndim == 1:
            full_ecg = full_ecg[np.newaxis, :]
        elif full_ecg.ndim == 2 and full_ecg.shape[0] > 1:
            # If we have multiple leads, let's use the second lead (index 1)
            full_ecg = full_ecg[1:2, :]
        
        x = full_ecg  # This should now be shape (1, length)
        y = self.Y[i]

        return torch.from_numpy(x).float(), y

        return torch.from_numpy(x).float(), y

    @staticmethod
    def collate_fn(batch):
        # This function can be used to handle variable-length sequences if needed
        x, y = zip(*batch)
        x = [torch.Tensor(x_) for x_ in x]
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0)
        y = torch.stack(y)
        return x, y


class ecg_dataset_for_inference(torch.utils.data.Dataset):
    """ Pytorch dataloader for pre-loaded ecg data inference """
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


def get_dataset_from_configs(data_cfg, preprocess_cfg, dataset_idx=None, split_idx=None, sanity_check=False):
    """ load ecg dataset from config files, extracting only the 2nd lead """
    if data_cfg.data is not None:
        # For inference mode
        x = preprocess_signal(data_cfg.data, preprocess_cfg, get_sample_rate(data_cfg.header))
        x = x[1:2, :]  # Extract only the 2nd lead
        y = np.zeros((1), np.float32)  # dummy label for code compatibility
        X = [torch.from_numpy(x).float()]
        Y = [torch.from_numpy(y)]
        dataset = ecg_dataset_for_inference(X, Y)
    else:
        # For training/validation mode
        if data_cfg.filenames is not None:
            filenames_all = data_cfg.filenames
        else:
            filenames_all = get_filenames_from_split(data_cfg, dataset_idx, split_idx)
        
        print(f"Total files found: {len(filenames_all)}")
        w=0
        filenames, sample_rates, Y = [], [], []
        for filename in filenames_all:
            if sanity_check:
                break
            try:
                
                header = load_header(filename)
                # print(header)
                y = preprocess_label(get_labels(header), data_cfg.scored_classes, data_cfg.equivalent_classes)
                # print(y)
                if np.sum(y) != 0 or (split_idx == "train" and preprocess_cfg.all_negative):
                    print(w)
                    filenames.append(filename)
                    sample_rates.append(get_sample_rate(header))
                    Y.append(torch.from_numpy(y))
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
        
        print(f"Files accepted: {len(filenames)}")
        
        if not filenames:
            print("No valid files found. Here's some debug information:")
            
            raise ValueError("No valid files found. Check your data directory and configurations.")
        
        dataset = ecg_dataset(filenames, sample_rates, Y, preprocess_cfg)
    
    print(f"Dataset size: {len(dataset)}")
    return dataset


def load_data(filename):
    """ load data from WFDB files """
    data = np.asarray(loadmat(filename)['val'], dtype=np.float32)
    
    # Ensure data is 2D
    if data.ndim == 1:
        data = data[np.newaxis, :]
    elif data.ndim > 2:
        data = data.squeeze()
        if data.ndim == 1:
            data = data[np.newaxis, :]
    
    return data


def load_header(filename):
    """ load header from WFDB files """
    HEADER = open(filename + ".hea", 'r')
    header = HEADER.readlines()
    HEADER.close()

    return header


def get_labels(header):
    """ get labels from header """
    labels = []
    for line in header:
        if line.lower().startswith('# dx:'):  # Case-insensitive check, allowing for space
            # Split on ':' and strip whitespace
            dx_part = line.split(':', 1)[1].strip()
            # Split on comma if multiple labels, otherwise use the whole string
            labels = [label.strip() for label in dx_part.split(',') if label.strip()]
            break  # Stop after finding the Dx line
    
    # print(f"Labels found: {labels}")  # Debug print
    return labels


def get_sample_rate(header):
    """ get sample frequency from header """
    sample_rate = int(header[0].strip().split()[2])

    return sample_rate


def get_filenames_from_split(data_cfg, fold, split_idx, base_path):
    filenames_all = []
    dataset_dir_mapping = {
        "cpsc_2018": "cpsc_2018",
        "cpsc_2018_extra": "cpsc_2018_extra",
        "st_petersburg_incart": "st_petersburg_incart",
        "ptb": "ptb",
        "ptb-xl": "ptb-xl",
        "georgia": "georgia"
    }
    print(f"Base path: {base_path}")
    print(f"Datasets in data_cfg: {data_cfg.datasets}")

    for dataset, dir_name in dataset_dir_mapping.items():
        dataset_path = os.path.join(base_path, dir_name)
        print(f"Checking dataset path: {dataset_path}")
        if not os.path.exists(dataset_path):
            print(f"  Directory does not exist: {dataset_path}")
            continue

        if dataset not in data_cfg.split:
            print(f"  Dataset {dataset} not in data_cfg.split")
            continue

        split_data = data_cfg.split[dataset]
        print(f"  Split data for {dataset}: {split_data.keys()}")

        filenames = []
        if fold in list(range(10)):
            if split_idx == "train":
                for i in range(10):
                    if i != fold:
                        fold_key = f'fold{i}'
                        if fold_key in split_data:
                            filenames.extend([os.path.join(dataset_path, f) for f in split_data[fold_key]])
            elif split_idx == "val":
                fold_key = f'fold{fold}'
                if fold_key in split_data:
                    filenames.extend([os.path.join(dataset_path, f) for f in split_data[fold_key]])
        elif fold == "sanity":
            filenames.extend([os.path.join(dataset_path, f) for f in split_data.get("all", [])[:1]])
        else:
            filenames.extend([os.path.join(dataset_path, f) for f in split_data.get(split_idx, [])])

        valid_filenames = []
        for filename in filenames:
            if os.path.exists(filename):
                valid_filenames.append(filename)
            elif os.path.exists(filename + ".mat"):
                valid_filenames.append(filename + ".mat")

        filenames_all.extend(valid_filenames)

    return filenames_all

def get_eval_dataset_split_idxs(data_cfg):
    """ get dataset and split idxs for evaluation """
    dataset_split_idxs = []

    dataset_idxs = data_cfg.datasets + ["all"] if len(data_cfg.datasets) > 1 else data_cfg.datasets
    for dataset_idx in dataset_idxs:
        if data_cfg.fold in list(range(10)):
            dataset_split_idxs.append(dataset_idx + "_" + "val")
        elif data_cfg.fold in ["all", "sanity"]:
            dataset_split_idxs.append(dataset_idx + "_" + data_cfg.fold)
        else:
            if "val" in data_cfg.split[data_cfg.datasets[0]]:
                dataset_split_idxs.append(dataset_idx + "_" + "val")
            dataset_split_idxs.append(dataset_idx + "_" + "test")

    return dataset_split_idxs


def get_loss_weights_and_flags(data_cfg, run_cfg, dataset_train=None):
    """ get class and confusion weights for training """
    class_weight = np.ones(len(data_cfg.scored_classes), dtype=np.float32)
    if run_cfg.class_weight and dataset_train is not None:
        Y = np.stack(dataset_train.Y, 0)
        class_weight = np.sum(Y, axis=0).astype(np.float32)
        class_weight[class_weight == 0] = 1
        max_num = np.max(class_weight)
        for i in range(len(class_weight)):
            class_weight[i] = np.sqrt(max_num / class_weight[i])

    confusion_weight_flag = run_cfg.confusion_weight
    confusion_weight = load_weights(data_cfg.path + "weights.csv", data_cfg.scored_classes)

    return class_weight, confusion_weight, confusion_weight_flag


def collate_into_list(args):
    """ collate variable-length ecg signals into list """
    X = [a[0] for a in args]
    Y = torch.stack([a[1] for a in args], 0)
    return X, Y


def collate_into_block(batch, l, stride):
    """
    collate variable-length ecg signals into block
    for those longer than chunk_length, divide them into l-point chunks with (overlapping) stride
    """
    X, Y = batch
    if stride is None:
        # assume all ecg signals have same length
        X_block = torch.stack(X, 0)
        X_flag  = X[0].new_ones((len(X)), dtype=torch.bool)
    else:
        # collate variable-length ecg signals
        b, c, = 0, X[0].shape[0]
        for x in X:
            b += int(np.ceil((x.shape[1] - l) / float(stride) + 1))

        X_block = X[0].new_zeros((b, c, l))
        X_flag  = X[0].new_zeros((b), dtype=torch.bool)
        idx = 0
        for x in X:
            num_chunks = int(np.ceil((x.shape[1] - l) / float(stride) + 1))
            for i in range(num_chunks):
                if i != num_chunks - 1:
                    X_block[idx] = x[:, i*stride:i*stride + l]
                    X_flag[idx] = False
                elif x.shape[1] > l:
                    X_block[idx] = x[:, -l:]
                    X_flag[idx] = True
                else:
                    X_block[idx, :, :x.shape[1]] = x[:, :]
                    X_flag[idx] = True
                idx += 1

    return X_block, X_flag, Y
