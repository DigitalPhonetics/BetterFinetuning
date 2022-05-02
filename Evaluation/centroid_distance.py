# Centriod Distance
import argparse
import pickle

import numpy as np
import pandas as pd
from pandas.core import arrays
from scipy.spatial import distance

parser = argparse.ArgumentParser(
    description="Centriod Distance Evaluation (Quality Analysis)"
)
parser.add_argument("embed_type", type=str, metavar="N", help="")
parser.add_argument("label_name", type=str, metavar="N", help="")
arg = parser.parse_args()

# TODO: label class automatically update based on label name
noise_lable_transfer = {
    "Clean": "Clean",
    "Babble": "Noisy",
    "Telephone": "Noisy",
    "Music": "Noisy",
}
age_lable_transfer = {
    60: 50,
    70: 50,
}


def filter(data_dict):
    filter_dict = data_dict.copy()
    for key in data_dict:
        emo_list = data_dict[key]["emotion"]
        if len(emo_list) != 0:
            filter_dict[key]["emotion"] = emo_list[0]
        else:
            del filter_dict[key]
    return filter_dict


class CentriodDistance:
    """
    1. Caluate centriod of each class/cluster
    2. Euclidan distance between each data point in a cluster to its respective cluster centroid
    3. put in pandas data frame
    Args:
        embed_type: 'contrastive', 'barlowtwins', or 'combined'
        label_name: 'valence_arousal', 'age', 'gender', 'noise'
        label_classes: based on label name
        train_result_path: path to trained embedding result
        test_result_path: path to test embedding result (we evaluate this frist)
    """

    def __init__(self, embed_type, label_name, train_result_path, test_result_path):
        self.embed_type = embed_type
        self.label_name = label_name
        self.train_result_path = train_result_path
        self.test_result_path = test_result_path

    def extract_embedding_label(self, embed_type, label_name, file_path):
        """
        return embeddings and corresponding labels
        """
        with open(file_path, "rb") as fp:
            data = pickle.load(fp)
            if self.label_name == "emotion":
                filter(data)
            df_data = pd.DataFrame.from_dict(data, orient="index")
            df_data = df_data[[embed_type, label_name]]
            df_data = df_data.dropna(subset=[embed_type])
        return df_data

    def _get_centriod_distance(self, vectors: arrays, centriod: arrays) -> float:

        """
        Given all vectors of one class and its centriod point, return average centroid distance.
        """
        vectors = vectors.tolist()
        avg_euclidean = np.zeros(len(vectors))
        for vector in vectors:
            euclidean_dist = distance.euclidean(vector, centriod)
            avg_euclidean = np.append(avg_euclidean, euclidean_dist)
        avg_euclidean = np.mean(avg_euclidean, axis=0)
        return avg_euclidean

    def centriod_distances_per_class(self):

        """
        Calculate the centroid given embeddings(a list of 1D vector)
        Euclidan distance between each data point in a cluster to
        its respective cluster centroid, put in pandas df
        """
        train_centriod_distances = {}
        test_centriod_distances = {}
        # data type of embedding: numpy.ndarray here
        train_df_data = self.extract_embedding_label(
            embed_type=self.embed_type,
            label_name=self.label_name,
            file_path=self.train_result_path,
        )

        test_df_data = self.extract_embedding_label(
            embed_type=self.embed_type,
            label_name=self.label_name,
            file_path=self.test_result_path,
        )

        """
        if self.label_name == "noise_type":
            train_df_data["noise_type"] = train_df_data["noise_type"].replace(
                noise_lable_transfer
            )
            test_df_data["noise_type"] = test_df_data["noise_type"].replace(
                noise_lable_transfer
            )
        """

        if self.label_name == "age":
            train_df_data["age"] = train_df_data["age"].replace(age_lable_transfer)
            test_df_data["age"] = test_df_data["age"].replace(age_lable_transfer)

        label_classes = train_df_data[self.label_name].unique().tolist()

        # put all in pandas data strcture, so we can extract embedding base on label
        for label in label_classes:
            # get centroid/mean of vectors from each class, and calculate the average euclidean distance between it and all datapoint

            # training set
            train_one_class = train_df_data[train_df_data[self.label_name] == label]
            one_class_vectors = train_one_class[self.embed_type].to_numpy()
            one_class_centriod = one_class_vectors.mean(axis=0)
            avg_centriod_distance = self._get_centriod_distance(
                one_class_vectors, one_class_centriod
            )
            train_centriod_distances[label] = [
                avg_centriod_distance,
                len(train_one_class),
            ]

            # test set
            test_one_class = test_df_data[test_df_data[self.label_name] == label]
            test_one_class_vectors = test_one_class[self.embed_type].to_numpy()
            test_one_class_centriod = test_one_class_vectors.mean(axis=0)
            test_avg_centriod_distance = self._get_centriod_distance(
                test_one_class_vectors, test_one_class_centriod
            )
            test_centriod_distances[label] = [
                test_avg_centriod_distance,
                len(test_one_class),
            ]

        # return result in the pandas format
        train_centriod_distances = pd.DataFrame.from_dict(
            train_centriod_distances,
            orient="index",
            columns=["centroid_distance", "datapoints"],
        )
        print(
            "--- Centriod Distance of {}/{} (train)--- \n{}".format(
                self.label_name, self.embed_type, train_centriod_distances
            )
        )

        test_centriod_distances = pd.DataFrame.from_dict(
            test_centriod_distances,
            orient="index",
            columns=["centroid_distance", "datapoints"],
        )
        print(
            "--- Centriod Distance of {}/{} (test)--- \n{}".format(
                self.label_name, self.embed_type, test_centriod_distances
            )
        )
        # TODO: write into csv file? (better copy-paste)
        return train_centriod_distances, test_centriod_distances


if __name__ == "__main__":
    embed_type = arg.embed_type
    label_name = arg.label_name

    # TODO: auto search path

    centriod_distance = CentriodDistance(
        embed_type=embed_type,
        label_name=label_name,
        # train_result_path=noise_combine_train,
        # test_result_path=noise_combine_test,
    )
    centriod_distance.centriod_distances_per_class()
