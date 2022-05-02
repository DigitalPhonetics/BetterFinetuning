import os
import pickle
import random
from pathlib import Path

import pandas as pd

# Relabel to make it has same label in TIMIT
AGE = {
    "teens": 10,
    "twenties": 20,
    "thirties": 30,
    "fourties": 40,
    "fifties": 50,
    "sixties": 60,
    "seventies": 70,
    "eighties": 80,
}


class CommonVoiceReader:
    def __init__(
        self, data_path, save_dir, rebuild_data_cache, reduce_data=True
    ) -> None:
        self.data_path = str(data_path)
        self.rebuild_data_cache = rebuild_data_cache
        self.save_dir = save_dir
        self.reduce_data = reduce_data

    def process(self):
        data = dict()
        # 1. check if data cache
        if (
            os.path.exists(os.path.join(self.save_dir, "CommonVoice_data_cache_1.pkl"))
            and not self.rebuild_data_cache
        ):
            with open(
                os.path.join(self.save_dir, "CommonVoice_data_cache.pkl"), "rb"
            ) as fp:
                data = pickle.load(fp)

            print("Prepared CommonVoice data.")
            return data, pd.DataFrame.from_dict(data, orient="index")

        # dev_df = pd.read_csv("{}/cv-valid-dev.csv".format(self.data_path))
        # test_df = pd.read_csv("{}/cv-valid-test.csv".format(self.data_path))
        train_df = pd.read_csv("{}/cv-valid-train.csv".format(self.data_path))

        ### We only need age and gender data
        # pdList = [dev_df, test_df, train_df]
        # all_df = pd.concat(pdList)
        all_df = train_df
        # all_df = all_df[all_df.accent != "us"]
        all_df = all_df[all_df.gender != "other"]
        all_df = all_df[all_df["age"].notna() & all_df["gender"].notna()]
        all_df.reset_index(inplace=True, drop=True)

        # we only use part of CommonVoice data
        if self.reduce_data:
            male = all_df.index[all_df["gender"] == "male"].tolist()
            m_n = int(len(male) * 0.88)
            female = all_df.index[all_df["gender"] == "female"].tolist()
            f_n = int(len(female) * 0.65)
            random_male = random.sample(male, m_n)
            random_female = random.sample(female, f_n)
            random_male.extend(random_female)
            all_df = all_df.drop(random_male)

        def full_datapath(filename):
            file = filename
            file = file.split("/")[0]
            return "{}/{}/{}".format(self.data_path, file, filename)

        all_df["filename"] = all_df["filename"].apply(full_datapath)

        for index, row in all_df.iterrows():
            try:
                p = Path(row["filename"])
                # x, fs = torchaudio.load(p)
                logid = p.stem
                data[logid] = {
                    "path": row["filename"],
                    "age": row["age"],
                    "gender": row["gender"],
                }
            except Exception as error:
                print(error)
                pass

        if (
            not os.path.exists(
                os.path.join(self.save_dir, "CommonVoice_data_cache.pkl")
            )
            or self.rebuild_data_cache
        ):
            with open(
                os.path.join(self.save_dir, "CommonVoice_data_cache.pkl"), "wb"
            ) as fp:
                pickle.dump(data, fp)
            print(
                "Saved CommonVoice data in {}".format(
                    self.save_dir + "/" + "CommonVoice_data_cache.pkl"
                )
            )

        print("Gender Statistics Distribution:\n", all_df["gender"].value_counts())
        print("Age Statistics Distribution:\n", all_df["age"].value_counts())

        return data, pd.DataFrame.from_dict(data, orient="index")
