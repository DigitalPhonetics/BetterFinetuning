"""

Read VOiCE Data: https://iqtlabs.github.io/voices/Lab41-SRI-VOiCES_README/
source-16k contains the original librispeech source audio.
distant-16k contains the VOiCES data
distractors : distractor noise recordings with no foreground audio for all rooms
room-response : recorded sound to determine room-response for all rooms
speech : for each room, recordings of foreground audio with babble, music, television or no distractor noise, arranged by speaker ID in each subfolder
Note that We only use speech directory.


"""
import os
import tqdm
import pickle
import glob
import torchaudio
import pandas as pd
from pathlib import Path

# distractor
NOISE_TYPE_DICT = {
    "none": "Clean",
    "babb": "Babble",
    "tele": "Telephone",
    "musi": "Music",
}


class VOiCESReader:
    def __init__(self, data_path, save_dir, rebuild_data_cache) -> None:
        self.data_path = str(data_path)
        self.rebuild_data_cache = rebuild_data_cache
        self.save_dir = save_dir

    def process(self) -> dict:
        data = dict()
        # 1. check if data cache
        if (
            os.path.exists(os.path.join(self.save_dir, "VOiCES_data_cache.pkl"))
            and not self.rebuild_data_cache
        ):
            with open(os.path.join(self.save_dir, "VOiCES_data_cache.pkl"), "rb") as fp:
                data = pickle.load(fp)
            print("Prepared VOiCES data.")
            return data, pd.DataFrame.from_dict(data, orient="index")
        speech_wav_files = glob.glob(
            "{}/distant-16k/speech/train/*/*/*/*.wav".format(self.data_path)
        )

        for path_filename in tqdm.tqdm(speech_wav_files):
            p = Path(path_filename)
            logid = p.stem
            # logid sample: Lab41-SRI-VOiCES-rm1-none-sp0083-ch003054-sg0005-mc01-stu-clo-dg090
            labels = logid.split("-")
            noise_type = labels[4]
            # skip broken audio file
            try:
                x, fs = torchaudio.load(p)
                data[logid] = {
                    "path": path_filename,
                    "noise": NOISE_TYPE_DICT[noise_type],
                }
            except Exception as error:
                print(error)
                pass

        # 5. build cache for existed data: save_dir
        # 6. condition for rebuild_data_dict
        if (
            not os.path.exists(os.path.join(self.save_dir, "VOiCES_data_cache.pkl"))
            or self.rebuild_data_cache
        ):
            with open(os.path.join(self.save_dir, "VOiCES_data_cache.pkl"), "wb") as fp:
                pickle.dump(data, fp)
            print(
                "Saved VOiCES data in {}".format(
                    self.save_dir + "/" + "VOiCES_data_cache.pkl"
                )
            )
        return data, pd.DataFrame.from_dict(data, orient="index")
