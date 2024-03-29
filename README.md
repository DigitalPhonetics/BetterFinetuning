# Better Finetuning

This is the Code accompanying our paper on finetuning self-supervised general speech representations with a combination
of contrastive and non-contrastive methods.

We propose a two-step finetuning framework that improves intermediate representations before feeding them into a
classifier. We demonstrate its success in a wide range of speech analysis tasks. Further we show an approach that
combines contrastive and non-contrastive losses and can explicitly improve cross-class variance and within-class
invariance in the embedding space, especially for some fuzzy speech properties. Additionally, to the best of our
knowledge, this work is the first that explores the use of Barlow Twins in the speech domain.

All the experiments and the training code are public and documented. The code and documentation for the data handling
pipeline will be added soon™

---

### Installation

1. clone repository and build virtual environment

```
git clone git@github.com:DigitalPhonetics/Representation-Adjustment.git
python3 -m venv venv
. venv/bin/activate
```

2. Install required packages

```
pip install -r requirements.txt
(alternative) pip install -e .
```

3. Select the pretrained-backbone of your choice in Training/config.py

wav2vec 2.0 and HuBERT are supported

---

### Data Preprocessing

```
# TBD: Will add the details on the datasets later
```

---

### Model Training

Example 1: Finetuning the contrastive model for speech emotion classification with 10 epochs. If there is an existing
checkpoint that you want to use, please change model_path in the script.

```
cd ContrastiveLearning/
python Siamese_train_pipeline.py emotion 10 model_path
```

Example 2: Finetuning the non-contrastive model for background sound classification with 100 epochs.

```
cd BarlowTwins/
python BarlowTwins_train_pipeline.py noise 100 model_path
```

---

### Evaluation

```
cd Evaluation/
```

Example 1: If you want to evaluate the centroid distance of gender embedding finetuned with the contrastive approach,
you will need to put the output embedding file within the repository, and specify the file path.

```
python centriod_distance.py contrastive gender file_path
```

Example 2: If you want to evaluate the Davies-Bouldin Index of emotion embedding finetuned with
non-contrastive/barlowtwins approach, you will need to put the output embedding file within the reposoitory, and specify
the file path.

```
python davies_bouldin.py barlowtwins emotion file_path
```
