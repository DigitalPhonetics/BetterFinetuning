import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import optim
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from ContrastiveLearning.CommonVoice_TIMIT_AgeDataset import (
    AgeDataset,
    generate_cv_timit_age_samples,
)
from ContrastiveLearning.CommonVoice_TIMIT_GenderDataset import (
    GenderDataset,
    generate_cv_timit_gender_samples,
)
from ContrastiveLearning.EmotionDataset import (
    UnifiedEmotionDataset,
    generate_emotion_samples,
)
from ContrastiveLearning.IEMOCAPDataset import IEMOCAPDataset, generate_iemocap_samples
from ContrastiveLearning.LibrispeechDataset import (
    LibrispeechDataset,
    generate_librispeech_samples,
)
from ContrastiveLearning.NoiseDataset import (
    VOiCESSiameseDataset,
    generate_noise_samples,
)
from ContrastiveLearning.models import TripletLossNet
from util import (
    save_model,
    save_objects,
)


def _get_cosine_distance(v1, v2):
    return 1 - F.cosine_similarity(v1, v2)


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    losses = []
    positive_accuracy = 0
    negative_accuracy = 0

    positive_distances = []
    negative_distances = []
    embedding_results = []

    for batch_idx, batch in enumerate(tqdm.tqdm(train_loader)):
        anchor, positive, negative, logid_a, logid_p, logid_n, label = batch
        anchor, positive, negative = (
            anchor.to(device),
            positive.to(device),
            negative.to(device),
        )
        # clear out the gradients of all Variables
        # in this optimizer
        optimizer.zero_grad()
        anchor_out, positive_out, negative_out = model(anchor, positive, negative)
        # Extract embedding intermediately from the model perform best
        embedding_results.append(
            [anchor_out, logid_a, positive_out, logid_p, negative_out, logid_n]
        )
        loss = model.loss(anchor_out, positive_out, negative_out)
        losses.append(loss.item())

        # disabled gradient calculation
        with torch.no_grad():
            p_distance = _get_cosine_distance(anchor_out, positive_out)
            positive_distances.append(torch.mean(p_distance).item())

            n_distance = _get_cosine_distance(anchor_out, negative_out)
            negative_distances.append(torch.mean(n_distance).item())

            positive_distance_mean = np.mean(positive_distances)
            negative_distance_mean = np.mean(negative_distances)

            positive_std = np.std(positive_distances)
            threshold = positive_distance_mean + 3 * positive_std
            # the threshold here can be adjusted

            positive_results = (
                p_distance < threshold
            )  # implies correct similarity/distance
            positive_accuracy += torch.sum(positive_results).item()

            negative_results = n_distance >= threshold  # Ture or False
            negative_accuracy += torch.sum(negative_results).item()

        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                "{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    time.ctime(time.time()),
                    epoch,
                    batch_idx * len(anchor),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            positive_distance_mean = np.mean(positive_distances)
            negative_distance_mean = np.mean(negative_distances)
            print(
                "Train Set: positive_distance_mean: {}, negative_distance_mean: {}, std: {}, threshold: {}".format(
                    positive_distance_mean,
                    negative_distance_mean,
                    positive_std,
                    threshold,
                )
            )

    positive_accuracy_mean = 100.0 * positive_accuracy / len(train_loader.dataset)
    negative_accuracy_mean = 100.0 * negative_accuracy / len(train_loader.dataset)
    return (
        np.mean(losses),
        positive_accuracy_mean,
        negative_accuracy_mean,
        embedding_results,
    )


def test(model, device, test_loader, log_interval=None):
    model.eval()
    losses = []
    positive_accuracy = 0
    negative_accuracy = 0

    postitive_distances = []
    negative_distances = []
    embedding_results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm.tqdm(test_loader)):
            with autocast():
                anchor, positive, negative, logid_a, logid_p, logid_n, label = batch
                anchor, positive, negative = (
                    anchor.to(device),
                    positive.to(device),
                    negative.to(device),
                )
                a_out, p_out, n_out = model(anchor, positive, negative)

                embedding_results.append(
                    [a_out, logid_a, p_out, logid_p, n_out, logid_n]
                )

                test_loss_on = model.loss(a_out, p_out, n_out, reduction="mean").item()
                losses.append(test_loss_on)

                p_distance = _get_cosine_distance(a_out, p_out)
                postitive_distances.append(torch.mean(p_distance).item())

                n_distance = _get_cosine_distance(a_out, n_out)
                negative_distances.append(torch.mean(n_distance).item())

                positive_distance_mean = np.mean(postitive_distances)
                negative_distance_mean = np.mean(negative_distances)

                positive_std = np.std(postitive_distances)
                threshold = positive_distance_mean + 3 * positive_std

                positive_results = p_distance < threshold
                positive_accuracy += torch.sum(positive_results).item()

                negative_results = n_distance >= threshold
                negative_accuracy += torch.sum(negative_results).item()

                if log_interval is not None and batch_idx % log_interval == 0:
                    print(
                        "{} Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            time.ctime(time.time()),
                            batch_idx * len(anchor),
                            len(test_loader.dataset),
                            100.0 * batch_idx / len(test_loader),
                            test_loss_on,
                        )
                    )

    test_loss = np.mean(losses)
    positive_accuracy_mean = 100.0 * positive_accuracy / len(test_loader.dataset)
    negative_accuracy_mean = 100.0 * negative_accuracy / len(test_loader.dataset)

    positive_distance_mean = np.mean(postitive_distances)
    negative_distance_mean = np.mean(negative_distances)
    print(
        "Test Set: positive_distance_mean: {}, negative_distance_mean: {}, std: {}, threshold: {}".format(
            positive_distance_mean, negative_distance_mean, positive_std, threshold
        )
    )

    print(
        "\nTest set: Average loss: {:.4f}, Positive Accuracy: {}/{} ({:.0f}%),  Negative Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            positive_accuracy,
            len(test_loader.dataset),
            positive_accuracy_mean,
            negative_accuracy,
            len(test_loader.dataset),
            negative_accuracy_mean,
        )
    )
    return test_loss, positive_accuracy_mean, negative_accuracy_mean, embedding_results


def main(
    input_path,
    speech_property,
    batch_size,
    output_size,
    num_epoches,
    rebuild_dataloaders_cache=False,
    use_cuda=True,
):
    cache_dir = "/mount/arbeitsdaten/synthesis/chenci/thesis_prosody_embedding/contrastive_dataloader_cache"

    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using Device: ", device)

    if speech_property == "Noise":
        # Train and Test dataloader
        model_path = "/mount/arbeitsdaten/synthesis/chenci/thesis_prosody_embedding/models/noise/"
        margin = 1.0
        learn_rate = 0.0001
        decay = 0.005
        (
            train_CLEAN,
            train_Babble,
            train_Telephone,
            train_Music,
            test_CLEAN,
            test_Babble,
            test_Telephone,
            test_Music,
        ) = generate_noise_samples(input_path)
        TrainDataset = VOiCESSiameseDataset(
            cache_dir=cache_dir,
            CLEAN_list=train_CLEAN,
            NOISE_list_1=train_Babble,
            NOISE_list_2=train_Telephone,
            NOISE_list_3=train_Music,
            rebuild_cache=rebuild_dataloaders_cache,
            phrase="train",
        )

        TestDataset = VOiCESSiameseDataset(
            cache_dir=cache_dir,
            CLEAN_list=test_CLEAN,
            NOISE_list_1=test_Babble,
            NOISE_list_2=test_Telephone,
            NOISE_list_3=test_Music,
            rebuild_cache=rebuild_dataloaders_cache,
            phrase="test",
        )

    if speech_property == "Emotion":
        model_path = "/mount/arbeitsdaten/synthesis/chenci/thesis_prosody_embedding/models/emotion/"
        margin = 1.0
        learn_rate = 0.0001
        decay = 0.0001
        train_emotion, test_emotion = generate_emotion_samples(input_path)

        TrainDataset = UnifiedEmotionDataset(
            cache_dir=cache_dir,
            VALENCE_AROUSAL=train_emotion,
            loading_processes=16,
            rebuild_cache=rebuild_dataloaders_cache,
            phrase="train",
        )

        TestDataset = UnifiedEmotionDataset(
            cache_dir=cache_dir,
            VALENCE_AROUSAL=test_emotion,
            loading_processes=8,
            rebuild_cache=rebuild_dataloaders_cache,
            phrase="test",
        )

    if speech_property == "IEMOCAP":
        model_path = "/mount/arbeitsdaten/synthesis/chenci/thesis_prosody_embedding/models/iemocap/"
        margin = 1.0
        learn_rate = 0.0001
        decay = 0.0001
        train_emotion, test_emotion = generate_iemocap_samples(input_path)

        TrainDataset = IEMOCAPDataset(
            cache_dir=cache_dir,
            VALENCE_AROUSAL=train_emotion,
            loading_processes=16,
            rebuild_cache=rebuild_dataloaders_cache,
            phrase="train",
        )

        TestDataset = IEMOCAPDataset(
            cache_dir=cache_dir,
            VALENCE_AROUSAL=test_emotion,
            loading_processes=8,
            rebuild_cache=rebuild_dataloaders_cache,
            phrase="test",
        )

    if speech_property == "Gender":
        model_path = "/mount/arbeitsdaten/synthesis/chenci/thesis_prosody_embedding/models/gender/"
        cv_path = "/mount/arbeitsdaten/synthesis/chenci/thesis_prosody_embedding/wav2vec_output/CommonVoice_Dataset_enc_and_transformer_mean.pkl"
        timit_path = "/mount/arbeitsdaten/synthesis/chenci/thesis_prosody_embedding/wav2vec_output/TIMIT_enc_and_transformer_mean.pkl"

        # parameter
        margin = 1.0
        learn_rate = 0.0001
        decay = 0.001

        # train_male, train_female, test_male, test_female = generate_cv_gender_samples(cv_path)
        (
            train_male,
            train_female,
            test_male,
            test_female,
        ) = generate_cv_timit_gender_samples(cv_path, timit_path)

        TrainDataset = GenderDataset(
            cache_dir=cache_dir,
            male_samples=train_male,
            female_samples=train_female,
            rebuild_cache=rebuild_dataloaders_cache,
            phrase="train",
        )

        TestDataset = GenderDataset(
            cache_dir=cache_dir,
            male_samples=test_male,
            female_samples=test_female,
            rebuild_cache=rebuild_dataloaders_cache,
            phrase="test",
        )

    if speech_property == "Age":
        # In order to compare with relavant work, we fine-tuned on CommonVoice and evaluate on TIMIT

        model_path = (
            "/mount/arbeitsdaten/synthesis/chenci/thesis_prosody_embedding/models/age/"
        )
        cv_path = "/mount/arbeitsdaten/synthesis/chenci/thesis_prosody_embedding/wav2vec_output/CommonVoice_Dataset_enc_and_transformer_mean.pkl"
        timit_path = "/mount/arbeitsdaten/synthesis/chenci/thesis_prosody_embedding/wav2vec_output/TIMIT_enc_and_transformer_mean.pkl"

        # parameter
        margin = 1.2
        learn_rate = 0.0001
        decay = 0.001

        train_age, test_age = generate_cv_timit_age_samples(
            cv_path=cv_path, timit_path=timit_path
        )
        # train_age, test_age = generate_cv_age_samples(cv_path)

        TrainDataset = AgeDataset(
            cache_dir=cache_dir,
            age_samples=train_age,
            rebuild_cache=rebuild_dataloaders_cache,
            phrase="train",
        )

        TestDataset = AgeDataset(
            cache_dir=cache_dir,
            age_samples=test_age,
            rebuild_cache=rebuild_dataloaders_cache,
            phrase="test",
        )

    if speech_property == "speaker_librispeech":
        model_path = (
            "/mount/arbeitsdaten/synthesis/chenci/thesis_prosody_embedding/models/SID/"
        )
        margin = 1.0
        learn_rate = 0.0001
        decay = 0.0001

        train_data, test_data = generate_librispeech_samples(input_path)

        TrainDataset = LibrispeechDataset(
            cache_dir=cache_dir,
            INPUT=train_data,
            loading_processes=16,
            rebuild_cache=rebuild_dataloaders_cache,
            phrase="train",
        )

        TestDataset = LibrispeechDataset(
            cache_dir=cache_dir,
            INPUT=test_data,
            loading_processes=8,
            rebuild_cache=rebuild_dataloaders_cache,
            phrase="test",
        )

    train_loader = DataLoader(
        batch_size=batch_size,
        dataset=TrainDataset,
        drop_last=True,
        num_workers=8,
        pin_memory=False,
        shuffle=True,
        prefetch_factor=8,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        batch_size=batch_size,
        dataset=TestDataset,
        drop_last=True,
        num_workers=8,
        pin_memory=False,
        shuffle=True,
        prefetch_factor=8,
        persistent_workers=True,
    )

    model = TripletLossNet(output_size=output_size, margin=margin, device=device).to(
        device
    )
    # TODO: add it back later
    """
    model = restore_model(model, model_path)
    (
        last_epoch,
        max_accuracy,
        train_losses,
        test_losses,
        train_positive_accuracies,
        train_negative_accuracies,
        test_positive_accuracies,
        test_negative_accuracies,
    ) = restore_objects(model_path, (0, 0, [], [], [], [], [], []))
    """
    (
        max_accuracy,
        train_losses,
        test_losses,
        train_positive_accuracies,
        train_negative_accuracies,
        test_positive_accuracies,
        test_negative_accuracies,
    ) = (0, [], [], [], [], [], [])

    train_embed_results = []
    test_embed_results = []

    # start = last_epoch + 1 if max_accuracy > 0 else 0
    optimizer = optim.Adam(
        model.parameters(), lr=learn_rate, weight_decay=decay
    )  # L2 Regularization

    for epoch in range(num_epoches + 1):
        (
            train_loss,
            train_positive_accuracy,
            train_negative_accuracy,
            train_embeddings,
        ) = train(model, device, train_loader, optimizer, epoch, 500)
        (
            test_loss,
            test_positive_accuracy,
            test_negative_accuracy,
            test_embeddings,
        ) = test(model, device, test_loader)

        print(
            "After epoch: {}, train loss is : {}, test loss is: {} \n"
            "train positive accuracy: {}, train negative accuracy: {} \n"
            "test positive accuracy: {}, and test negative accuracy: {} \n".format(
                epoch,
                train_loss,
                test_loss,
                train_positive_accuracy,
                train_negative_accuracy,
                test_positive_accuracy,
                test_negative_accuracy,
            )
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_positive_accuracies.append(train_positive_accuracy)
        test_positive_accuracies.append(test_positive_accuracy)

        train_negative_accuracies.append(train_negative_accuracy)
        test_negative_accuracies.append(test_negative_accuracy)

        test_accuracy = (test_positive_accuracy + test_negative_accuracy) / 2

        # we only store model that has best test performance during this training time
        # we need to have store and restore model/object corresponding to each speech properties
        if test_accuracy > max_accuracy:
            # we extract the best embedding
            # TODO: also save to save_objects, otherwise it will be empty list when test_accuracy < max_accuracy
            train_embed_results = train_embeddings
            test_embed_results = test_embeddings

            max_accuracy = test_accuracy
            save_model(model, epoch, out_path=model_path)
            save_objects(
                (
                    epoch,
                    max_accuracy,
                    train_losses,
                    test_losses,
                    train_positive_accuracies,
                    train_negative_accuracies,
                    test_positive_accuracies,
                    test_negative_accuracies,
                ),
                epoch,
                out_path=model_path,
            )
            print("saved epoch: {} as checkpoint".format(epoch))
        else:
            print("This model does not perform well, so we abandon it...")

    return train_embed_results, test_embed_results


# Extract embedding in the last epoches, should remove later
# TODO: store train and test embedding at once
# TODO: use builded model and thenfeature_extractor
def extract_embedding(old_dict_path, file_name, embeddings, new_dict_save_path):
    with open(old_dict_path, "rb") as fp:
        dict_info = pickle.load(fp)
    for idx, batch in enumerate(embeddings):
        anchor, positive, negative, logid_a, logid_p, logid_n = batch
        for idx, (a, p, n, log_a, log_p, log_n) in enumerate(
            zip(anchor, positive, negative, logid_a, logid_p, logid_n)
        ):
            # convert back to numpy array so we can load data in cup-only machine
            a = a.detach().cpu().numpy()
            # p = p.detach().cpu().numpy()
            # n = n.detach().cpu().numpy()
            dict_info[log_a].update({"contrastive_a": a})
            # for SID task
            # new_dict[log_a] = {
            #   "contrastive_a": a,
            #    "speaker_id": dict_info[log_a]["speaker_id"]
            # }

            # print(dict_info[log_a])
            print("suceess!")
            # dict_info[log_p].update({"contrastive_p": p})
            # dict_info[log_n].update({"contrastive_n": n})

    with open(
        os.path.join(
            new_dict_save_path, "{}_contrastive_embedding.pkl".format(file_name)
        ),
        "wb",
    ) as fp:
        pickle.dump(dict_info, fp)
    print("Saved embedding to {}".format(new_dict_save_path))
