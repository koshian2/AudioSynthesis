import requests
import time
import json
import os
import numpy as np
import wave
import glob

def _audio_query(text, filename, max_retry):
    query_payload = {"text": text,
                     "speaker": 0}
    for query_i in range(max_retry):
        r = requests.post("http://localhost:50021/audio_query", 
                        params=query_payload, timeout=(3.0, 300.0))
        if r.status_code == 200:
            query_data = r.json()
            break
        time.sleep(1)
    else:
        raise ConnectionError("リトライ回数が上限に到達しました。 audio_query : ", filename, "/", text, r.text)
    return query_data, query_i

def _audio_synthesis(query_data, filename, max_retry):
    # synthesis
    synth_payload = {"speaker": 0}
    for synth_i in range(max_retry):
        query_data["intonationScale"] = 1.2
        query_data["volumeScale"] = 2.5
        query_data["prePhonemeLength"] = 0.5
        query_data["postPhonemeLength"] = 0.5
        r = requests.post("http://localhost:50021/synthesis", params=synth_payload, 
                          data=json.dumps(query_data), timeout=(3.0, 300.0))
        if r.status_code == 200:
            binary = r.content
            break
        time.sleep(1)
    else:
        raise ConnectionError("リトライ回数が上限に到達しました。 syntesis : ", filename, "/", r.text)
    return binary, synth_i

def get_mora_length(query_data):
    # moraの長さ
    moras = []
    for phrases in query_data["accent_phrases"]:
        moras += phrases["moras"]
        if phrases["pause_mora"]:
            moras.append(phrases["pause_mora"])
    return len(moras)

def voicevox_synthesis(text, filename, max_retry=5):
    raw_query, query_i = _audio_query(text, filename, max_retry)

    # 150字以上は分割
    if len(text) > 200:
        sentences = text.split("。")
        if len(sentences) == 1:
            return None # 文字数が長すぎて1文しかない場合は
        #length_counter = np.cumsum(len(x)+1 for x in sentences])
        #idx_t = np.argmin(length_counter > len(text)//2)
        idx_t = len(sentences)//2
        splited_text = ["。".join(sentences[:idx_t]), "。".join(sentences[idx_t:])]
        print(splited_text)
        actual_query = []
        for s in splited_text:
            x = _audio_query(s, filename, max_retry)
            actual_query.append(x[0])
            query_i += x[1]
    else:
        actual_query = [raw_query]

    audio_binaries, synth_i = [], 0
    for q in actual_query:
        x = _audio_synthesis(q, filename, max_retry)
        # 分割していない場合はそのまま保存
        if len(actual_query) == 1:
            with open(filename, "wb") as fp:
                fp.write(x[0])
        # 分割した場合はリストに追加
        else:
            audio_binaries.append(x[0])
        synth_i += x[1]

    # 分割した場合の結合
    if len(audio_binaries) >= 2:
        audio_binaries = [np.frombuffer(x, np.int16) for x in audio_binaries]
        print([x.shape for x in audio_binaries])
        x = np.concatenate(audio_binaries)
        print([x.shape for x in audio_binaries])
        with wave.Wave_write(filename) as fp:
            fp.setframerate(24000)
            fp.setnchannels(1)
            fp.setsampwidth(2)
            fp.writeframes(x.tobytes())

    print(f"{filename} は query={query_i+1}回, synthesis={synth_i+1}回で正常に保存されました")
    return raw_query

def render_datasets(article_data_path):
    audio_dir = "data/pretrain_dataset/audio"
    moras_dir = "data/pretrain_dataset/moras"
    if not os.path.isdir(audio_dir):
        os.makedirs(audio_dir)
    if not os.path.isdir(moras_dir):
        os.makedirs(moras_dir)
    failed = []

    with open(article_data_path, encoding="utf-8") as fp:
        data = fp.read().split("\n")
    # VOICEVOXを再起動するたびに合成できなかったファイルで進行が遅くなるのを回避
    synthesised_file = sorted(glob.glob(moras_dir+"/*.json"))
    if len(synthesised_file) > 0:
        last_index = len(synthesised_file) # 結構いい加減なやり方
    else:
        last_index = 0
    print("Start from ", last_index)

    for i, item in enumerate(data):
        #if i < last_index:
        #    continue
        x = item.split("\t")
        audio_path = f"{audio_dir}/{x[0]}.wav"
        moras_path = f"{moras_dir}/{x[0]}.json"
        if os.path.exists(moras_path):
            continue
        
        try:
            moras = voicevox_synthesis(x[1].strip(), audio_path)
        except:
            failed.append(x[0])
        else:
            with open(moras_path, "w", encoding="utf-8") as fp:
                json.dump(moras, fp, ensure_ascii=False, indent=4, separators=(',', ': '))
    with open("data/pretrain_dataset/logs.txt", "w", encoding="utf-8") as fp:
        fp.write("\n".join(failed))

if __name__ == "__main__":
    render_datasets("data/pretrain_dataset/crawl/articles_all/hokkaido.txt")
