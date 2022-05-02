# Davies-Bouldin Index
import argparse
import pickle

import pandas as pd
from sklearn.metrics import davies_bouldin_score

parser = argparse.ArgumentParser(
    description="Davies-Bouldin Index Evaluation (Quality Analysis)"
)
parser.add_argument("embed_type", type=str, metavar="N", help="")
parser.add_argument("label_name", type=str, metavar="N", help="")
arg = parser.parse_args()

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


class DaviesBouldinIndex:
    """
    1. extract all classes we have
    2. extract list of n_features-dimensional data points
    3. Predicted labels for each sample -> we need to trasnfer label into float
    4. the resulting Davies-Bouldin score ()
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

    def get_davies_bouldin_index(self):
        """
        Calculate the centroid given embeddings(a list of 1D vector)
        Euclidan distance between each data point in a cluster to
        its respective cluster centroid, put in pandas df
        """
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

        if self.label_name == "age":
            train_df_data["age"] = train_df_data["age"].replace(age_lable_transfer)
            test_df_data["age"] = test_df_data["age"].replace(age_lable_transfer)
        """
        unique_labels_list = train_df_data[self.label_name].unique().tolist()
        label2index = {label: float(ix) for ix, label in enumerate(unique_labels_list)}

        train_labels_list = train_df_data[self.label_name].tolist()
        train_labels_list = [label2index[label] for label in train_labels_list]
        embeddings_list = train_df_data[self.embed_type].tolist()
        train_embeddings_list = [embedding for embedding in embeddings_list]

        test_labels_list = test_df_data[self.label_name].tolist()
        test_labels_list = [label2index[label] for label in test_labels_list]
        embeddings_list = test_df_data[self.embed_type].tolist()
        test_embeddings_list = [embedding for embedding in embeddings_list]

        train_db = davies_bouldin_score(train_embeddings_list, train_labels_list)
        test_db = davies_bouldin_score(test_embeddings_list, test_labels_list)

        print(
            "--- Davies-Bouldin Index of {}/{} (train)--- \n{}".format(
                self.label_name, self.embed_type, train_db
            )
        )

        print(
            "--- Davies-Bouldin Index of {}/{} (test)--- \n{}".format(
                self.label_name, self.embed_type, test_db
            )
        )
        return train_db, test_db


if __name__ == "__main__":
    embed_type = arg.embed_type
    label_name = arg.label_name

    # TODO: auto search path

    db = DaviesBouldinIndex(
        embed_type=embed_type,
        label_name=label_name,
        # train_result_path=age_combine_train,
        # test_result_path=age_combine_test,
    )
    db.get_davies_bouldin_index()
