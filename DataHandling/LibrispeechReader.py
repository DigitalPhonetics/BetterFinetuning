import glob
import os
import pickle
from pathlib import Path

import pandas as pd
import torchaudio
import tqdm


class LibrispeechReader:
    def __init__(self, data_path, save_dir, rebuild_data_cache) -> None:
        self.data_path = str(data_path)
        self.rebuild_data_cache = rebuild_data_cache
        self.save_dir = save_dir

    def extract_data(self, speech_wav_files) -> dict:
        data = {}
        for path_filename in tqdm.tqdm(speech_wav_files):
            p = Path(path_filename)
            logid = p.stem
            # logid example: Lab41-SRI-VOiCES-rm1-none-sp0083-ch003054-sg0005-mc01-stu-clo-dg090
            labels = logid.split("_")
            speaker_id = labels[0]

            # skip broken audio file
            try:
                x, fs = torchaudio.load(p)
                data[logid] = {
                    "speaker_id": speaker_id,
                    "path": path_filename,
                }
            except Exception as error:
                if error:
                    pass
        return data

    def process(self) -> dict:
        train_file_name = "LibriSpeech_train_data_cache.pkl"
        test_file_name = "LibriSpeech_test_data_cache.pkl"

        # 1. check if data cache exist and if rebuild_data_cache is True
        if (
                os.path.exists(os.path.join(self.save_dir, train_file_name))
                and not self.rebuild_data_cache
        ):
            with open(os.path.join(self.save_dir, train_file_name), "rb") as fp:
                train_data = pickle.load(fp)
            with open(os.path.join(self.save_dir, test_file_name), "rb") as fp:
                test_data = pickle.load(fp)

            print("Prepared LibriSpeech training & testing data.")
            train_df = pd.DataFrame.from_dict(train_data, orient="index")
            test_df = pd.DataFrame.from_dict(test_data, orient="index")

            return train_data, train_df, test_data, test_df

        # Process data
        train_wav_files = glob.glob(
            "{}/train-clean-360/*/*/*.wav".format(self.data_path)
        )
        test_wav_files = glob.glob("{}/test-clean/*/*/*.wav".format(self.data_path))

        train_data = self.extract_data(train_wav_files)
        test_data = self.extract_data(test_wav_files)

        train_df = pd.DataFrame.from_dict(train_data, orient="index")
        test_df = pd.DataFrame.from_dict(test_data, orient="index")

        if (
                not os.path.exists(os.path.join(self.save_dir, train_file_name))
                or self.rebuild_data_cache
        ):
            with open(os.path.join(self.save_dir, train_file_name), "wb") as fp:
                pickle.dump(train_data, fp)
            with open(os.path.join(self.save_dir, test_file_name), "wb") as fp:
                pickle.dump(test_data, fp)
            print(
                "Saved LibriSpeech train data in {}".format(
                    self.save_dir + "/" + train_file_name
                )
            )
            print(
                "Saved LibriSpeech test data in {}".format(
                    self.save_dir + "/" + test_file_name
                )
            )
        print("Speaker ID in training and testing set: ")
        print(train_df["speaker_id"].value_counts())
        print(test_df["speaker_id"].value_counts())

        return train_data, train_df, test_data, test_df
