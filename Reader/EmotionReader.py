"""
Used Dataset: IEMOCAP(en), EMODB(de), Emotional_Singapore(zh), korean(ko)
Unified Label: anger, neutral (boredom),  happiness, sadness

* we also have valence and arousal label for all datasets
"""

import glob
import json
import os
from collections import Counter
from itertools import chain
from pathlib import Path

import pandas as pd
import torchaudio
import tqdm

VALENCE_AROUSAL = {
    "happy": "+,+",
    "angry": "-,+",
    "sad": "-,n",
    "neutral": "n,n",
    "fear": "-,+",
    "disgust": "-,n",
    "surprise": "n,+",
}
data = dict()


class EmotionReader:
    def __init__(self, dataset_name, data_path, save_dir, rebuild_data_cache) -> None:
        self.dataset_name = str(dataset_name)
        self.data_path = str(data_path)
        self.rebuild_data_cache = rebuild_data_cache
        self.save_dir = save_dir

    @staticmethod
    def emotiontts_open_db_label(filenames_list_1, filenames_list_2):
        data_1 = {}
        data_2 = {}
        emo_lookup_1 = {
            "0": "neutral",
            "1": "happy",
            "2": "angry",
            "3": "sad",
        }  # emotional-to-emotional
        emo_lookup_2 = {
            "0": "neutral",
            "1": "angry",
            "2": "happy",
            "3": "sad",
        }  # plain-to-emotional

        for filename in filenames_list_1:
            p = Path(filename)
            logid = p.stem
            emotion_label = logid[
                5
            ]  # nea00001 -> 0 or 1 or 2 or 3 (general/happy/anger/sad)
            data_1[logid] = {
                "emotion": emo_lookup_1[emotion_label],
            }
        for filename in filenames_list_2:
            p = Path(filename)
            logid = p.stem
            emotion_label = logid[
                5
            ]  # nea00001 -> 0 or 1 or 2 or 3 (general/happy/anger/sad)
            data_2[logid] = {
                "emotion": emo_lookup_2[emotion_label],
            }
        data = {**data_1, **data_2}
        return data

    @staticmethod
    def EMODB_label(filenames_list):
        data = {}
        emo_lookup = {
            "A": "fear",
            "W": "angry",
            "L": "neutral",
            "N": "neutral",
            "E": "disgust",
            "F": "happy",
            "T": "sad",
        }
        # remove anxiety/fear, make disgust as anger, boredom(L) as neutral
        for filename in filenames_list:
            p = Path(filename)
            logid = p.stem
            # 03a01Fa
            emotion_label = logid[5]
            data[logid] = {
                "emotion": emo_lookup[emotion_label],
            }
        return data

    def process(self) -> dict:
        data = dict()
        wav_files = list()

        cache_file_name = self.dataset_name + "_data_cache.json"
        cache_file_path = os.path.join(
            self.save_dir, cache_file_name
        )  # Turn repeated commend into cache_file_path

        if os.path.exists(cache_file_path) and not self.rebuild_data_cache:
            with open(cache_file_path, "r") as fp:
                data = json.load(fp)
            print("Prepared {} data.".format(self.dataset_name))
            return data

        if self.dataset_name == "Korean_Read_Speech_Corpus":
            # wave files
            wav_files = glob.glob("{}/dataset/*/*.wav".format(self.data_path))
            label_path = "/mount/arbeitsdaten/synthesis/chenci/Datasets/Korean-Read-Speech-Corpus/dataset/Korean_Read_Speech_Corpus_sample.json"
            with open(label_path, "r") as fp:
                labels = json.load(fp)
            airbnb = labels["AirbnbStudio"]
            anechoic = labels["AnechoicChamber"]
            dance = labels["DanceStudio"]
            ks_labels = {**airbnb, **anechoic, **dance}

            # read label file
            data = self.korean_read_speech_label(ks_labels)

        if self.dataset_name == "emotiontts_open_db":
            # wave files
            # /Dataset/SpeechCorpus/Emotional/emotional-to-emotional/nea/wav
            wav_files_1 = glob.glob(
                "{}/Dataset/SpeechCorpus/Emotional/emotional-to-emotional/*/wav/*.wav".format(
                    self.data_path
                )
            )
            wav_files_2 = glob.glob(
                "{}/Dataset/SpeechCorpus/Emotional/plain-to-emotional/*/wav/*.wav".format(
                    self.data_path
                )
            )
            # read label file
            wav_files = glob.glob(
                "{}/Dataset/SpeechCorpus/Emotional/*/*/wav/*.wav".format(self.data_path)
            )
            data = self.emotiontts_open_db_label(wav_files_1, wav_files_2)

        if self.dataset_name == "EMO-DB":
            # /mount/arbeitsdaten/synthesis/chenci/Datasets/EMO-DB(de)/wav
            wav_files = glob.glob("{}/wav/*.wav".format(self.data_path))
            data = self.EMODB_label(wav_files)

        if self.dataset_name == "Emotional_Speech_Dataset_Singapore":
            data = {}
            # /mount/arbeitsdaten/synthesis/chenci/Datasets/Emotional_Speech_Dataset_Singapore/0001/Happy
            happy_wav = glob.glob("{}/*/Happy/*.wav".format(self.data_path))
            angry_wav = glob.glob("{}/*/Angry/*.wav".format(self.data_path))
            neutral_wav = glob.glob("{}/*/Neutral/*.wav".format(self.data_path))
            sad_wav = glob.glob("{}/*/Sad/*.wav".format(self.data_path))
            surprise_wav = glob.glob("{}/*/Surprise/*.wav".format(self.data_path))
            emotion_idx = {
                0: "neutral",
                1: "happy",
                2: "angry",
                3: "sad",
                4: "surprise",
            }
            wav_files = [neutral_wav, happy_wav, angry_wav, sad_wav, surprise_wav]
            for idx, wav_file in enumerate(wav_files):
                for wav in wav_file:
                    p = Path(wav)
                    logid = p.stem
                    data[logid] = {
                        "path": wav_file,
                        "emotion": emotion_idx[idx],
                    }
            wav_files = list(
                chain(neutral_wav, happy_wav, angry_wav, sad_wav, surprise_wav)
            )

        for path_filename in tqdm.tqdm(wav_files):
            p = Path(path_filename)
            logid = p.stem
            x, fs = torchaudio.load(p)
            # Extract other signal feature here
            data[logid].update(
                {
                    "path": path_filename,
                }
            )

        if not os.path.exists(cache_file_path) or self.rebuild_data_cache:
            with open(cache_file_path, "w") as fp:
                json.dump(data, fp)
            print(
                "Saved {} data in {}".format(
                    self.dataset_name, self.save_dir + "/" + cache_file_name
                )
            )
        return data


# noqa: C901
class Concatenate:
    def __init__(self, Korean, German, Mandrine, English, save_dir):

        self.ko_dict = Korean
        self.de_dict = German
        self.zh_dict = Mandrine
        self.en_dict = English
        self.save_dir = save_dir

    def concatenate(self):
        def relabel(input_dict):
            rebuild_dict = {}

            for key in input_dict:
                emotion = input_dict[key]["emotion"]
                # for IEMOCAP
                if isinstance(emotion, list):
                    if not emotion:
                        # print("find empty list!")
                        continue
                    else:
                        emotion = emotion[
                            0
                        ]  # if the audio has multiple emotion, we choose the main emotion

                rebuild_dict[key] = {
                    "filename": input_dict[key]["filename"],
                    "emotion": emotion,
                    "valence_arousal": VALENCE_AROUSAL[emotion],
                }

            return rebuild_dict

        rebuild_dict_korean = relabel(self.ko_dict)
        rebuild_dict_german = relabel(self.de_dict)
        rebuild_dict_mandrine = relabel(self.zh_dict)
        rebuild_dict_english = self.en_dict  # relabel("English", self.en_dict)

        relabeled_united_data = {
            **rebuild_dict_korean,
            **rebuild_dict_german,
            **rebuild_dict_mandrine,
            **rebuild_dict_english,
        }

        # count all valence/arousal class we have:
        all_v_a = [value["valence_arousal"] for value in relabeled_united_data.values()]
        count_result = Counter(all_v_a)
        print("Emotion data statistics('valence, arousal'): {}".format(count_result))

        # put into panda dataframe

        save_path = os.path.join(self.save_dir, "relabl_unified_emotion_data.json")
        with open(save_path, "w") as fp:
            json.dump(relabeled_united_data, fp)
        print("save relabel unified emotion data into {}".format(save_path))
        df_data = pd.DataFrame.from_dict(relabeled_united_data, orient="index")

        return relabeled_united_data, df_data
