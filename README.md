# SpeechRepresentationFinetuning
****We are releasing the source code, but will keep update the repository until in the end of May.****

Code accompanying our paper on finetuning self-supervised general speech representations with a combination of contrastive and non-contrastive methods.

We propose a novel two-step finetuning framework that improves intermediate representations
before feeding them into a classifier and demonstrates its success in a wide range of speech
analysis tasks. We demonstrate an innovative approach that combines contrastive and non-contrastive losses
can explicitly improve class variance and invariance in the embedding space, especially for
some fuzzy speech properties. Additionally, to the best of our knowledge, this work is the first that explores the use of Barlow Twins
in the speech domain.

---
### Installation
1. clone repository and build virtual environment
```
git clone git@github.com:DigitalPhonetics/Representation-Adjustment.git
python3 -m venv venv
. venv/bin/activate
```

2. Install required package
```
pip install -r requirements.txt
(alternative) pip install -e .
```
---

### Date Preprocessing 
```
# TBD: Will add the detail of dataset later
```
---
### Model Training
```
# For example, finetuning the contrastive model for speech emotion classification with 10 epoches
cd ContrastiveLearning/Siamese_train_pipeline.py
python Siamese_train_pipeline.py emotion 10
```

```
# For example, finetuning the non-contrastive model for background sound classification with 10 epoches
cd BarlowTwins/BarlowTwins_train_pipeline.py
python BarlowTwins_train_pipeline.py noise 10
```

---
### Evaluation
```
# If you want to evaluate the centroid distance of gender embedding finetuned with contrastive approach
# You will need to put the output embedding file within the reposotory 
cd Evaluation/centroid_distance.py
python centriod_distance.py contrastive gender file_path
```
```
# If you want to evaluate the Davies-Bouldin Index of emotion embedding finetuned with non-contrastive/barlowtwins approach
# You will need to put the output embedding file within the reposotory, and specify the file path
cd Evaluation/davies_bouldin.py
python davies_bouldin.py barlowtwins emotion file_path
```