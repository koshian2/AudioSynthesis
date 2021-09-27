import glob
import numpy as np
import json
from attrdict import AttrDict
import os
import librosa
from numpy.core.fromnumeric import sort
from pandas.core import base
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def pack_npy(h, file_key):
    audio_dir = "data/pretrain_dataset/audio"
    moras_dir = "data/pretrain_dataset/moras"
    write_dir = "data/pretrain_dataset/npy"
    if not os.path.exists(f"{audio_dir}/{file_key}.wav"):
        return
    if os.path.exists(f"{write_dir}/{file_key}.npz"):
        return

    phoneme_list = [
        "pau",
        "A",
        "E",
        "I",
        "N",
        "O",
        "U",
        "a",
        "b",
        "by",
        "ch",
        "cl",
        "d",
        "dy",
        "e",
        "f",
        "g",
        "gw",
        "gy",
        "h",
        "hy",
        "i",
        "j",
        "k",
        "kw",
        "ky",
        "m",
        "my",
        "n",
        "ny",
        "o",
        "p",
        "py",
        "r",
        "ry",
        "s",
        "sh",
        "t",
        "ts",
        "ty",
        "u",
        "v",
        "w",
        "y",
        "z",
    ]
    with open(f"{moras_dir}/{file_key}.json", encoding="utf-8") as fp:
        moras = json.load(fp)
    phoneme_indices = []
    phoneme_length = []
    pitches = []

    def parse_mora(mora):
        con_idx = phoneme_list.index(mora["consonant"]) if mora["consonant"] else -1
        vow_idx = phoneme_list.index(mora["vowel"]) if mora["vowel"] else -1
        phoneme_indices.append([con_idx, vow_idx])
        # length
        x = [
            mora["consonant_length"] if mora["consonant_length"] is not None else 0.0,
            mora["vowel_length"] if mora["vowel_length"] is not None else 0.0
        ]
        phoneme_length.append(x)
        # pitches
        pitches.append(mora["pitch"])

    for phrases in moras["accent_phrases"]:
        for mora in phrases["moras"]:
            parse_mora(mora)
        if phrases["pause_mora"]:
            parse_mora(phrases["pause_mora"])

    # 縦に音素、横に時間軸、値はピッチ
    upsample_rates = np.cumprod([[1,] + list(h.coarse_model.upsample_rates) 
                                + list(h.fine_model.upsample_rates)])  
    conditional_inputs_flag = np.array([True,] + list(h.coarse_model.conditional_norms) + 
                                        list(h.fine_model.conditional_norms))
    frame_rates = h.sampling_rate / upsample_rates[-1] * upsample_rates
    frame_rates = frame_rates[conditional_inputs_flag]
    unit_lengths = 1.0 / frame_rates # 1マスあたりの基準秒数

    # エンコード
    encode_data = []
    P = len(phoneme_list)
    for unit_len in unit_lengths:
        item = []
        # 開始時の無音データ
        item.append(np.zeros((P, int(h.silence_start/unit_len)), np.float32))
        for mora_idx, mora_lens, pitch in zip(phoneme_indices, phoneme_length, pitches):
            # 母音と子音
            for idx, length in zip(mora_idx, mora_lens):
                t = max(int(length / unit_len), 1)
                x = np.zeros((P, t), np.float32)
                x[idx, :] = pitch
                item.append(x)
        # 終了時の無音データ
        item.append(np.zeros((P, int(h.silence_end/unit_len)), np.float32))
        encode_data.append(np.concatenate(item, axis=1))

    # wave
    sound, sr = librosa.load(f"{audio_dir}/{file_key}.wav", sr=None) # sr=Noneをしないと22.05kHzにサンプリングされて-1～1のレンジから外れる

    # mel
    mel = librosa.feature.melspectrogram(y=sound,
                                        sr=sr,
                                        n_mels=h.num_mels,
                                        n_fft=h.n_fft,
                                        win_length=h.win_size,
                                        hop_length=h.hop_size)
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel/80.0) + 1.0 # logmel [0-1]
    sound = np.expand_dims(sound, axis=0)

    # write npz
    if not os.path.isdir(write_dir):
        os.makedirs(write_dir)
    x = {
        "sound": sound,
        "filename": file_key,
        "logmel": mel
    }
    for i, z in enumerate(encode_data):
        x[f"mora_{i}"] = z
    np.savez(f"{write_dir}/{file_key}", **x)

def check_max_moras():
    with open("data/pretrain_dataset/crawl/articles_all/hokkaido.txt", encoding="utf-8") as fp:
        data = fp.read()
    mora_lengths = []
    for line in tqdm(data.split("\n")):
        key, _ = line.split("\t")
        path = f"data/pretrain_dataset/moras/{key}.json"
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fp:
            query_data = json.load(fp)
        moras = []
        for phrases in query_data["accent_phrases"]:
            moras += phrases["moras"]
            if phrases["pause_mora"]:
                moras.append(phrases["pause_mora"])
        mora_lengths.append(len(moras))
    print(np.max(mora_lengths))
    print(np.quantile(mora_lengths, np.arange(11)/10))
    # 393がMAX


def pack_data():
    with open("config.json") as fp:
        h = AttrDict(json.load(fp))
    with open("data/pretrain_dataset/crawl/articles_all/hokkaido.txt", encoding="utf-8") as fp:
        data = fp.read()
    for line in tqdm(data.split("\n")):
        key, _ = line.split("\t")
        pack_npy(h, key)

def split_traintest():
    base_path = "data/pretrain_dataset"
    files = [os.path.basename(x) for x in sorted(glob.glob(f"{base_path}/npy/*.npz"))]
    train, test = train_test_split(files, random_state=1234, test_size=0.06)
    print(len(train), len(test))
    for p, t in zip(["train", "test"], [train, test]):
        with open(f"{base_path}/{p}.txt", "w", encoding="utf-8") as fp:
            fp.write("\n".join(sorted(t))) 

if __name__ == "__main__":
    pack_data()
