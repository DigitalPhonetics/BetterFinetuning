import glob
import json
import os
import re
from pathlib import Path

import pandas as pd
import tqdm

FRAMES_PER_SEC = 100
AV_DICT = {True: "+", False: "-"}
# we only neutral, sadness, anger, and happiness
# in order to compare with the result in "Emotion Recognition from Speech UsingWav2vec 2.0 Embeddings"
EMOTIONS_LIST = [
    "neutral",
    "state",
    "anger",
    "sadness",
    "fear",
    "happiness",
    "excited",
    "frustration",
    "annoyed",
    "worried",
    "melancolic",
    "indifferent",
]
RELABEL = {
    "neutral": "neutral",
    "anger": "anger",
    "fear": "sadness",
    "excited": "happiness",
    "sadness": "sadness",
    "frustration": "sadness",
    "happiness": "happiness",
    "annoyed": "anger",
    "worried": "sadness",
    "melancolic": "sadness",
    "indifferent": "neutral",
}


class IEMOCAPCategoryReader:
    def __init__(self, data_path, save_dir, rebuild_cache) -> None:
        self.data_path = str(data_path)
        self.min_duration = 0.0
        self.min_sad_frames_duration = 0.0
        self.compute_speech_rate = True
        self.sample = None
        self.rebuild_cache = rebuild_cache
        self.save_dir = save_dir

    def process(self):
        data = dict()
        att = list()
        emo = list()
        emotion_attribute = dict()

        if (
            os.path.exists(
                os.path.join(self.save_dir, "IEMOCAP_categorical_data_cache.json")
            )
            and not self.rebuild_cache
        ):
            with open(
                os.path.join(self.save_dir, "IEMOCAP_categorical_data_cache.json"), "r"
            ) as fp:
                data = json.load(fp)
            print("Prepared IEMOCAP data.")
            return data, pd.DataFrame.from_dict(data, orient="index")
        # list contain all wav file path
        sentences_wav_files = glob.glob(
            "{}/*/sentences/wav/*/*.wav".format(self.data_path)
        )
        # 1. add basic info of wav file
        for path_filename in tqdm.tqdm(sentences_wav_files):
            p = Path(path_filename)
            logid = p.stem
            # logid is name of wave file
            data[logid] = {
                "path": path_filename,
                "wavfile": p.name,
            }

        dialog_emoeval_attribute_files = glob.glob(
            "{}/*/dialog/EmoEvaluation/Attribute/*.txt".format(self.data_path)
        )
        dialog_emoeval_emotion_files = glob.glob(
            "{}/*/dialog/EmoEvaluation/Categorical/*.txt".format(self.data_path)
        )
        # 2. add activation, valence, dominance label
        for path_filename in tqdm.tqdm(dialog_emoeval_attribute_files):
            with open(path_filename, "r") as fp:
                att_texts = [line.rstrip() for line in fp]
                att_texts = [re.sub(r"[^\w\s]", "", att) for att in att_texts]
            att.extend(att_texts)

        # 3. add emtions label
        for path_filename in tqdm.tqdm(dialog_emoeval_emotion_files):
            with open(path_filename, "r") as fp:
                emo_texts = [line.rstrip() for line in fp]
                emo_texts = [re.sub(r"[^\w\s]", "", emo) for emo in emo_texts]
            emo.extend(emo_texts)

        # 4. add/convert attribute and emotion to data dict
        # here we might have multiple emo
        for emo_text in emo:
            emotion_info = emo_text.split(" ")
            # we only consider certain emotions
            emotion_info_filter = [
                emo for emo in emotion_info if emo.lower() in EMOTIONS_LIST
            ]

            # print(emotion_info) # ['Ses03M_impro04_M013', 'Anger', '']
            # print(emotion_info_filter) # ['Anger']
            emotion = [RELABEL[emo.lower()] for emo in emotion_info_filter]  # no logid

            # emotion_info[0] is logid
            if len(emotion) == 0:
                continue
            # might have multiple emotions, so we only choose the main emotion
            emotion_attribute[emotion_info[0]] = emotion[0]

        for att_text in att:
            att_text = att_text.split(" ")
            if len(att_text) >= 7:
                activation = float(att_text[2])
                valence = float(att_text[4])
                # dominance = float(att_text[6])
                emotion_pair = "{},{}".format(
                    AV_DICT[activation > 3], AV_DICT[valence > 3]
                )

                if activation == 3 and valence == 3:
                    emotion_pair = "n,n"

            # logid example : Ses01F_script02_2_M003
            try:
                data[att_text[0]].update(
                    {
                        "emotion": emotion_attribute[
                            att_text[0]
                        ],  # att_text[0] is logid
                        # "valence_raw": valence,
                        # "activation_raw": activation,
                        # "dominance_raw": dominance,
                        "valence_arousal": emotion_pair,
                        # "valence": emotion_pair[2],
                        # "arousal": emotion_pair[0],
                    }
                )
            except KeyError:
                continue

        if (
            not os.path.exists(
                os.path.join(self.save_dir, "IEMOCAP_categorical_data_cache.json")
            )
            or self.rebuild_cache
        ):
            with open(
                os.path.join(self.save_dir, "IEMOCAP_categorical_data_cache.json"), "w"
            ) as fp:
                json.dump(data, fp)
            print(
                "Saved IEMOCAP data in {}".format(
                    self.save_dir + "/" + "IEMOCAP_categorical_data_cache.json"
                )
            )
        data_df = pd.DataFrame.from_dict(data, orient="index")
        print("Emotion Statistics Distribution:\n", data_df["emotion"].value_counts())

        return data, data_df
