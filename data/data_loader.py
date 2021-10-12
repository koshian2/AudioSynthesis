import numpy as np
from torch.utils.data import Dataset
import pickle
import random
from data.pretrain_dataset import mora_phrase
import sys
sys.modules['mora_phrase'] = mora_phrase

class PretrainedDataset(Dataset):
    def __init__(self, h, split_type):
        super().__init__()
        self.h = h
        self.split_type = split_type

        with open(f"{self.h.dataset_dir}/{split_type}.txt", encoding="utf-8") as fp:
            self.pkl_keys = fp.read().split("\n")
        self.items = None

    def _cache_items(self):
        self.items = []
        for f in self.pkl_keys:
            with open(f"{self.h.dataset_dir}/{self.h.pickle_dir}/{f}", "rb") as fp:
                x = pickle.load(fp)
            #x = np.load(f"{self.h.dataset_dir}/{self.h.pickle_dir}/{f}", allow_pickle=True)
            self.items.append(x)

    def __len__(self):
        return len(self.pkl_keys)

    def __getitem__(self, idx):
        # cache to memory
        if self.items is None:
            self._cache_items()
        return self.items[idx] 

    def collate_fn(self, batches):
        result = []
        for sentences in batches:
            use_sentences = []
            if self.split_type == "train":
                if len(sentences) >= self.h.training_sentences:
                    indices = random.sample(range(len(sentences)), self.h.training_sentences)
                    for i in indices:
                        use_sentences.append(sentences[i])
                else:
                    use_sentences = sentences
            else:
                use_sentences = sentences
            result.append(use_sentences)
        return result

