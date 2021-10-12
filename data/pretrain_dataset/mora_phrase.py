import json
import librosa
import numpy as np

MORA_LIST = [
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

class Sentence:
    def __init__(self) -> None:
        self.texts = []
        self.mora_char = []
        self.mora_indices = []
        self.mora_length = []
        self.is_vowel = []
        self.pitch = []

        # エンコード用の疎行列の係数
        self.sparse_mora_indices = []
        self.sparse_time_indices = []
        self.sparse_duration = []
        self.sparse_values = []
        self.sparse_size = ()

        self.sound_data = None

        self.pause_length = 0.0
        self.pre_phoneme_length = 0.0
        self.post_phoneme_length = 0.0

        self.global_start_frame = 0

    def from_dict(self, mora_json):
        """VOICEVOXのaudio_queryから、センテンス単位のモーラ配列
            (accent_phrases以下の配列)

        Args:
            mora_json (list): audio_queryのモーラ配列
        """
        for phrases in mora_json:
            for mora in phrases["moras"]:
                self.texts.append(mora["text"])
                if mora["consonant"] is not None:
                    self.mora_char.append(mora["consonant"])

                    ind = MORA_LIST.index(mora["consonant"])
                    self.mora_indices.append(ind)

                    self.is_vowel.append(False)
                    self.mora_length.append(mora["consonant_length"])
                    self.pitch.append(mora["pitch"])

                if mora["vowel"] is not None:
                    self.mora_char.append(mora["vowel"])

                    ind = MORA_LIST.index(mora["vowel"])
                    self.mora_indices.append(ind)

                    self.is_vowel.append(True)
                    self.mora_length.append(mora["vowel_length"])
                    self.pitch.append(mora["pitch"])

            if phrases["pause_mora"] is not None:
                self.pause_length += phrases["pause_mora"]["vowel_length"]

    def __repr__(self) -> str:
        return str(self.__dict__ )+"\n"

    def quantize(self, sampling_rate, pre_phoneme_length, post_phoneme_length):
        cnt = int(pre_phoneme_length * sampling_rate)

        for idx, length, pitch in zip(
            self.mora_indices, self.mora_length, self.pitch):
            duration = max(int(length*sampling_rate), 1)

            self.sparse_mora_indices.append(idx)
            self.sparse_time_indices.append(cnt)
            self.sparse_duration.append(duration)
            self.sparse_values.append(pitch)

            cnt += duration
        
        cnt += int(self.pause_length * sampling_rate)
        cnt += int(post_phoneme_length * sampling_rate)

        self.sparse_size = (len(MORA_LIST), cnt)
        self.pre_phoneme_length = pre_phoneme_length
        self.post_phoneme_length = post_phoneme_length


    @staticmethod
    def from_audio_query(json_path, audio_path,
        sampling_rate=None,
        pre_phoneme_length=None,
        post_phoneme_length=None):
        """audio_queryを文単位の分割し、音素のインデックスを疎行列形式で記録し、
        分割された音声ファイルを埋め込みます。

        Args:
            json_path (str): audio_queryの内容が記録されたjsonのパス
            audio_path (str): wavファイルのパス
            sampling_rate (int, optional): サンプリングレート。
                Noneの場合はaudio_queryのサンプリングレートを参照。デフォルトはNone。
            pre_phoneme_length (float, optional): 読み上げ前の無音（秒）
                Noneの場合はaudio_queryのprePhonemeLengthを参照。デフォルトはNone。
            post_phoneme_length (float, optional): 読み上げ後の無音（秒）
                Noneの場合はaudio_queryのpostPhonemeLengthを参照。デフォルトはNone。

        Returns:
            List<Sentence>: アラインメントを取ったセンテンスの列
        """
        with open(json_path, encoding="utf-8") as fp:
            audio_query = json.load(fp)

        # dict parse
        sentences = []
        tmp_phrases = []
        for phrase in audio_query["accent_phrases"]:
            tmp_phrases.append(phrase)

            if phrase["pause_mora"] is not None:
                s = Sentence()
                s.from_dict(tmp_phrases)
                sentences.append(s)
                tmp_phrases = []
        else:
            if len(tmp_phrases) > 0:
                s = Sentence()
                s.from_dict(tmp_phrases)
                sentences.append(s)

        # quantize
        if sampling_rate is None:
            sampling_rate = audio_query["outputSamplingRate"]

        global_cnt = 0
        for i, s in enumerate(sentences):
            if pre_phoneme_length is None:
                empty_pre = audio_query["prePhonemeLength"] if i == 0 else 0.0
            else:
                empty_pre = pre_phoneme_length if i == 0 else 0.0

            if post_phoneme_length is None:
                empty_post = audio_query["postPhonemeLength"] if i == len(sentences) - 1 else 0.0
            else:
                empty_post = post_phoneme_length if i == len(sentences) - 1 else 0.0
                
            s.quantize(sampling_rate, empty_pre, empty_post)
            # Waveファイルをトリミングする開始フレーム
            s.global_start_frame = global_cnt
            global_cnt += s.sparse_size[1]

        # 音声をembed
        sound, sr = librosa.load(audio_path, sr=None) # float32 [-1, 1]

        # トリミング量
        trim_start = audio_query["prePhonemeLength"] - sentences[0].pre_phoneme_length
        trim_end = audio_query["postPhonemeLength"] - sentences[-1].post_phoneme_length
        trim_start, trim_end = int(trim_start*sampling_rate), int(trim_end*sampling_rate)

        if trim_start > 0:
            sound = sound[trim_start:]
        elif trim_start < 0:
            margin = np.zeros((np.abs(trim_start)), sound.dtype)
            sound = np.concatenate([margin, sound])

        if trim_end > 0:
            sound = sound[:-trim_end]
        elif trim_end < 0:
            margin = np.zeros((np.abs(trim_end)), sound.dtype)
            sound = np.concatenate([sound, margin])

        cnt = 0
        for s in sentences:
            s.sound_data = sound[cnt:cnt+s.sparse_size[1]]
            cnt += s.sparse_size[1]
        return sentences
