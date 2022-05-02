# Average Pariwsie Distance:


import argparse
import pickle

import pandas as pd
from scipy.spatial import distance

parser = argparse.ArgumentParser(
    description="Average Pariwsie Distance Evaluation (Quality Analysis)"
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


class PairwiseDistance:
    """
    1. extract all classes we have
    2. calculate the centroid(mean of all point belonged to the class) of all class
    3. calcuate the distance between this centroid and all other centroid, and average
    4. print average pairwise distnce for all class
    5. average all value to get final average
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

    def _get_avg_pairwise_dist(self, centroids) -> dict:

        """
        Given dictionary of label-centroid point, return label-average pairwise distance
        """
        result_dict = {}

        for label, centroid in centroids.items():
            centroids_cp = centroids.copy()
            pairwise_dist = 0.0
            del centroids_cp[label]
            for other_label, other_centroid in centroids_cp.items():
                dist = distance.euclidean(centroid, other_centroid)
                pairwise_dist += dist
            result_dict[label] = pairwise_dist  # / len(centroids)
        print(result_dict)

        return result_dict

    def avg_distances_per_class(self):

        """
        Calculate the centroid given embeddings(a list of 1D vector)
        Euclidan distance between each data point in a cluster to
        its respective cluster centroid, put in pandas df
        """
        train_pair_distances = {}
        test_pair_distances = {}
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

        if self.label_name == "noise_type":
            train_df_data["noise_type"] = train_df_data["noise_type"].replace(
                noise_lable_transfer
            )
            test_df_data["noise_type"] = test_df_data["noise_type"].replace(
                noise_lable_transfer
            )

        if self.label_name == "age":
            train_df_data["age"] = train_df_data["age"].replace(age_lable_transfer)
            test_df_data["age"] = test_df_data["age"].replace(age_lable_transfer)

        label_classes = train_df_data[self.label_name].unique().tolist()

        # put all in pandas data strcture, so we can extract embedding base on label
        for label in label_classes:
            # get centroid/mean of vectors from each class, and calculate the average euclidean distance between it and all datapoint

            # training set
            train_one_class = train_df_data[train_df_data[self.label_name] == label]
            # all vectors belong to this class
            one_class_vectors = train_one_class[self.embed_type].to_numpy()
            # get centroid of those vectors
            one_class_centriod = one_class_vectors.mean(axis=0)

            train_pair_distances[label] = one_class_centriod

            # test set
            test_one_class = test_df_data[test_df_data[self.label_name] == label]
            test_one_class_vectors = test_one_class[self.embed_type].to_numpy()
            test_one_class_centriod = test_one_class_vectors.mean(axis=0)

            test_pair_distances[label] = test_one_class_centriod

        # get average pairwise distance
        train_avg_distances = self._get_avg_pairwise_dist(train_pair_distances)
        test_avg_distances = self._get_avg_pairwise_dist(test_pair_distances)

        # return result in the pandas format
        train_avg_pair_distances = pd.DataFrame.from_dict(
            train_avg_distances,
            orient="index",
            columns=["avg_pairwise_dist"],
        )
        print(
            "--- Average Pariwsie Distance of {}/{} (train)--- \n{}".format(
                self.label_name, self.embed_type, train_avg_pair_distances
            )
        )

        test_avg_pair_distances = pd.DataFrame.from_dict(
            test_avg_distances,
            orient="index",
            columns=["avg_pairwise_dist"],
        )
        print(
            "--- Average Pariwsie Distance of {}/{} (test)--- \n{}".format(
                self.label_name, self.embed_type, test_avg_pair_distances
            )
        )
        # TODO: write into csv file? (better copy-paste)
        return train_avg_pair_distances, test_avg_pair_distances


if __name__ == "__main__":
    embed_type = arg.embed_type
    label_name = arg.label_name

    # TODO: auto search file in cache

    avg_pair_distance = PairwiseDistance(
        embed_type=embed_type,
        label_name=label_name,
        # train_result_path=iemocap_barlowtwins_train,
        # test_result_path=iemocap_barlowtwins_test,
    )
    avg_pair_distance.avg_distances_per_class()
