import glob
import os
import pickle
import torch


# it will delete redundant file
def _remove_files(files):
    for f in files:
        return os.remove(f)


def assert_dir_exits(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_model(model, epoch, out_path):
    assert_dir_exits(out_path)
    exist_chk_files = glob.glob("{}/*.pth".format(out_path))
    _remove_files(exist_chk_files)
    save_dir = out_path + str(epoch) + ".pth"
    torch.save(model.state_dict(), save_dir)
    print("Saved model in {}".format(save_dir))


"""
(epoch, max_accuracy, train_losses, test_losses, train_positive_accuracies,
train_negative_accuracies, test_positive_accuracies, test_negative_accuracies),
epoch, out_path=model_path
"""


def save_objects(obj, epoch, out_path):
    assert_dir_exits(out_path)
    exist_dat_files = glob.glob("{}/*.dat".format(out_path))
    _remove_files(exist_dat_files)
    # object should be tuple
    with open(out_path + str(epoch) + "siamese" + ".dat", "wb") as output:
        pickle.dump(obj, output)

    print("objects saved for epoch: {}".format(epoch))


def restore_model(model, out_path):
    chk_file = glob.glob(out_path + "*.pth")

    if chk_file:
        chk_file = str(chk_file[0])
        print("found model {}, restoring".format(chk_file))
        model.load_state_dict(torch.load(chk_file))
    else:
        print("Model not found, using untrained model")
    return model


def restore_objects(out_path, default):
    data_file = glob.glob(out_path + "*.dat")
    if data_file:
        data_file = str(data_file[0])
        print("found data {}, restoring".format(data_file))
        with open(data_file, "rb") as input_:
            obj = pickle.load(input_)
        return obj
    else:
        return default
