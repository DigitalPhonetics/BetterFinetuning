"""
Get Age & Gender Data from TIMIT

"""
import os
import tqdm
import json
import csv
import glob
import torchaudio
import pandas as pd
from pathlib import Path

# Relabel to make it has same label in CommonVoice

GENDER = {
    "M": "male",
    "F": "female",
}


class TIMITReader:
    def __init__(self, data_path, save_dir, rebuild_data_cache) -> None:
        self.data_path = str(data_path)
        self.rebuild_data_cache = rebuild_data_cache
        self.save_dir = save_dir

    def process(self) -> dict:
        data = dict()
        if (
            os.path.exists(os.path.join(self.save_dir, "TIMIT_data_cache.json"))
            and not self.rebuild_data_cache
        ):
            with open(os.path.join(self.save_dir, "TIMIT_data_cache.json"), "r") as fp:
                data = json.load(fp)
            print("Prepared TIMIT data.")
            return data
        speech_wav_files = glob.glob("{}/data/*/*/*/*.wav".format(self.data_path))
        speaker_info_df = pd.read_csv(
            os.path.join(self.data_path, "data_info_height_age.csv")
        )
        # speaker_info_df.set_index('ID', inplace=True)

        for path_filename in tqdm.tqdm(speech_wav_files):
            p = Path(path_filename)
            logid = p.stem
            match_g_id = path_filename.split("/")[-2]  # e.g., MCRC0
            match_id = match_g_id[1:]  # CRC0
            match_g = match_g_id[0]  # M
            # match_df = speaker_info_df["ID"]  # CRC0
            speaker_info = speaker_info_df.loc[speaker_info_df["ID"] == match_id]
            age = round(speaker_info["age"])
            age = (int(age * 10) // 100) * 10
            data[logid] = {"path": path_filename, "gender": GENDER[match_g], "age": age}
        data_df = pd.DataFrame.from_dict(data, orient="index")

        if (
            not os.path.exists(os.path.join(self.save_dir, "TIMIT_data_cache.json"))
            or self.rebuild_data_cache
        ):
            with open(os.path.join(self.save_dir, "TIMIT_data_cache.json"), "w") as fp:
                json.dump(data, fp)
            print(
                "Saved TIMIT data in {}".format(
                    self.save_dir + "/" + "TIMIT_data_cache.json"
                )
            )
        print("Gender Statistics Distribution:\n", data_df["gender"].value_counts())
        print("Age Statistics Distribution:\n", data_df["age"].value_counts())

        return data, data_df
