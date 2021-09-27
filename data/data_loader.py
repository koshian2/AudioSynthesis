import torch
import numpy as np
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, h, args, dataset_basedir, split_type):
        super().__init__()
        self.h = h
        self.args = args
        self.dataset_dir = dataset_basedir
        self.split_type = split_type

        with open(f"{dataset_basedir}/{split_type}.txt", encoding="utf-8") as fp:
            self.npy_keys = fp.read().split("\n")
        self.items = None

    def _cache_items(self):
        #print("Prepare data caching...", self.dataset_dir, self.split_type)
        self.items = []
        for f in self.npy_keys:
            x = np.load(f"{self.dataset_dir}/npy/{f}")
            self.items.append(x)
        #print("Caching completed. Load", len(self.items), "items.")

    def __len__(self):
        return len(self.npy_keys)

    def __getitem__(self, idx):
        # cache to memory
        if self.items is None:
            self._cache_items()
        return self.items[idx] 

    def collate_fn(self, batches):
        k_input_to_mel = np.prod(self.h.coarse_model.upsample_rates)
        k_input_to_wave = k_input_to_mel * np.prod(self.h.fine_model.upsample_rates)

        moras, y, y_mel = [], [], []
        for item in batches:
            m = []
            for k in range(self.h.mora_scales):
                m.append(item[f"mora_{k}"]) # 4個
            moras.append(m)
            if self.args.stage >= 2:
                y.append(item["sound"]) # (1, wave_length)
            y_mel.append(item["logmel"]) # (h.win_size, mel_size)

        # バッチできるようにPadしていく
        # 波形のPad
        batch_wave_period = max([x[0].shape[-1]*k_input_to_wave for x in moras]) # 入力モーラ列の2304倍
        if self.args.stage >= 2:
            for j in range(len(batches)):
                pad = np.zeros((1, batch_wave_period-y[j].shape[-1]), np.float32)
                y[j] = np.concatenate([y[j], pad], axis=-1)

        # モーラのPad
        batch_moras = []
        for k in range(self.h.mora_scales):
            batch_mora_period = max([moras[j][k].shape[-1] for j in range(len(batches))])
            if k == 0:
                batch_input_period = batch_mora_period
            item = []
            for j in range(len(batches)):
                pad = np.zeros((moras[j][k].shape[0],
                                batch_mora_period-moras[j][k].shape[-1]), np.float32)
                item.append(np.concatenate([moras[j][k], pad], axis=-1))
            batch_moras.append(torch.FloatTensor(item))
        for j in range(self.h.network_scales - self.h.mora_scales):
            batch_moras.append(None) # Conditional Normをしない層

        # メルスペクトログラム用のマスクとPad
        batch_mel_period = batch_input_period*k_input_to_mel
        mel_mask = np.zeros((len(batches), self.h.num_mels, batch_mel_period), np.float32)
        for j in range(len(batches)):
            mel_mask[j, :, np.arange(batch_mel_period)<y_mel[j].shape[1]] = 1.0
            pad = np.zeros((y_mel[j].shape[0],
                    batch_mel_period-y_mel[j].shape[1]), dtype=np.float32)
            y_mel[j] = np.concatenate([y_mel[j], pad], axis=-1) # 0-1

        if self.args.stage >= 2:
            return batch_moras, torch.FloatTensor(y), torch.FloatTensor(y_mel), torch.FloatTensor(mel_mask)
        else:
            return batch_moras, None, torch.FloatTensor(y_mel), torch.FloatTensor(mel_mask)

class PretrainedDataset(BaseDataset):
    def __init__(self, h, args, split_type):
        super().__init__(h, args, "data/pretrain_dataset", split_type)




if __name__ == "__main__":
    from test_code.dataloader_test import test_dataloader
    test_dataloader()

