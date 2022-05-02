import argparse
import pickle

import numpy as np
import pandas as pd

# TODO: build official predictor
# for now we use the result from training data and test data
# training data as label data and testing data as unlabel data
from scipy.spatial import distance
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description="One/Few-Shot Evaluation")
parser.add_argument("shot_numebr", type=int, metavar="N", help="")
parser.add_argument("embed_type", type=str, metavar="N", help="")
parser.add_argument("label_name", type=str, metavar="N", help="")
arg = parser.parse_args()

noise_lable_transfer = {
    "Clean": "Clean",
    "Babble": "Noisy",
    "Telephone": "Noisy",
    "Music": "Noisy",
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


class FewShotEvaluation:
    def __init__(
        self, shot_number, embed_type, label_name, train_result_path, test_result_path
    ):
        self.shot_number = shot_number
        self.embed_type = embed_type
        self.label_name = label_name

        self.train_result_path = train_result_path
        self.test_result_path = test_result_path

    # TODO: build a search function that find all directrionay and collect output file into list
    # and then we can automatically read file based on argument(speech type)

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
            df_data.dropna(subset=[embed_type], inplace=True)

        return df_data

    def _get_cosine_distance(self, v1, v2):
        """
        get
        depends on input type, if type is list, return average similarity
        """
        return distance.cosine(v1, v2)

    def _get_accuracy(self, y_ture_list, y_predict_list):
        """
        input: ture label list and predict label list
        return: accuracy score
        if binary case, we can calculate on TP, FP, TN, FN
        """
        accuracy = accuracy_score(y_ture_list, y_predict_list)
        return accuracy

    def predict_class(self):
        """
        one/few-shot evaluation depends on arugement: shot-num
        print evaluation score

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

        # Example: shot number = 3 -> {"+,+": [vector_1,vector_2,vector_3], "-,-": [vector_1,vector_2,vector_3], ...}
        label_classes = train_df_data[self.label_name].unique().tolist()
        predict_labels = []

        # for each test embedding
        for index, row in test_df_data.iterrows():
            # for each class
            cosine_distance = {}
            test_datapoint = row[self.embed_type]
            # print("test_datapoint:", test_datapoint)
            # print(len(test_datapoint))
            for label in label_classes:
                train_one_class = train_df_data[train_df_data[self.label_name] == label]
                # select shot number of trained embeddings
                # reference_datapoint = train_one_class[self.embed_type] # TODO: check if to_numpy() needed
                reference_datapoint = train_one_class.sample(n=self.shot_number)
                # print("reference_datapoint0:", reference_datapoint)
                reference_datapoints = np.zeros(len(test_datapoint))
                for index, row in reference_datapoint.iterrows():
                    reference_datapoint = row[self.embed_type]
                    # print("reference_datapoint1:", reference_datapoint)
                    reference_datapoints += reference_datapoint
                    # print("reference_datapoints:", reference_datapoints)
                reference_datapoint = reference_datapoints / self.shot_number
                # print("reference_datapoint3:", reference_datapoint)

                # calculate its similarity with 1/shot_num datapoint from each class
                cosine_distance[label] = self._get_cosine_distance(
                    reference_datapoint, test_datapoint
                )
            # select the min distance/high similarity
            predict_label = min(cosine_distance, key=cosine_distance.get)
            predict_labels.append(predict_label)

        test_ture_labels = test_df_data[self.label_name].tolist()
        """
        if self.label_name == "noise_type":
            predict_labels = [noise_lable_transfer[label] for label in predict_labels]
            test_ture_labels = [
                noise_lable_transfer[label] for label in test_ture_labels
            ]
        """
        # calculate acc
        accuracy = self._get_accuracy(test_ture_labels, predict_labels)
        print("-----Few Shot Classification ({}-shot)-----".format(self.shot_number))
        print(
            "Accuracy for {} embedding via {} approach is: {} ({} test datapoint)".format(
                self.label_name, self.embed_type, accuracy, len(test_df_data.index)
            )
        )
        # TODO: accuracy per class


if __name__ == "__main__":
    embed_type = arg.embed_type
    label_name = arg.label_name
    shot_number = arg.shot_numebr

    # TODO: auto search file in cache

    eval = FewShotEvaluation(
        shot_number=shot_number,
        embed_type=embed_type,
        label_name=label_name,
        # train_result_path=noise_combine_train,
        # test_result_path=noise_combine_test,
    )
    eval.predict_class()
