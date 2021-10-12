import os
import glob
from sklearn.model_selection import train_test_split
from mora_phrase import Sentence
from tqdm import tqdm
import pickle
import numpy as np

def split_traintest():
    base_path = "data/pretrain_dataset"
    files = [os.path.basename(x).replace(".wav", ".pkl") for x in sorted(glob.glob(f"{base_path}/audio/*.wav"))]

    train, test = train_test_split(files, random_state=1234, test_size=0.06)
    print(len(train), len(test))
    for p, t in zip(["train", "test"], [train, test]):
        with open(f"{base_path}/{p}.txt", "w", encoding="utf-8") as fp:
            fp.write("\n".join(sorted(t)))

def pack_pickle():
    moras_path = "data/pretrain_dataset/moras/*.json"
    audio_path = "data/pretrain_dataset/audio/*.wav"
    npy_path = "data/pretrain_dataset/pickle"
    file_list_path = "data/pretrain_dataset"

    moras = sorted(glob.glob(moras_path))
    audio = sorted(glob.glob(audio_path))

    def load_file_list(path):
        with open(f"{file_list_path}/{path}", encoding="utf-8") as fp:
            data = fp.read()
        return data.split("\n")

    train_list = load_file_list("train.txt")
    test_list = load_file_list("test.txt")

    for mora_file, audio_file in tqdm(zip(moras, audio)):
        key, _ = os.path.splitext(os.path.basename(mora_file))
        key += ".pkl"
        if key in train_list:
            pre_phoneme, post_phoneme = 0, 0
        elif key in test_list:
            pre_phoneme, post_phoneme = 0.5, 0.5
        else:
            raise ValueError("指定したファイルがtrainにもtestにもありません", key)

        sentences = Sentence.from_audio_query(mora_file, audio_file, 
            pre_phoneme_length=pre_phoneme, post_phoneme_length=post_phoneme)

        if not os.path.isdir(npy_path):
            os.makedirs(npy_path)

        #with open(f"{pickle_path}/{key}", "wb") as fp:
        #    pickle.dump(sentences, fp)

if __name__ == "__main__":
    huga()
