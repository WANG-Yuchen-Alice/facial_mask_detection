# facial_mask_detection

Repo for CS5242 Final Project [Group 10]: Neural Networks and Deep Learning in AY2022/23, Semester 1.

**Note**: This README is NOT the final report. This is only for instructions on our repo structure. Our final report is in the foder `./reports`.

### Repo Structure
- mlp/cnn/lstm/transformer: folders that contain the notebook/scripts/codes for each model, correspondingly
- reports: 
  - final report notebook
  - final presentation slides
  - project plan
- preparation: 
  - scripts for data collection, preprocessing, and augmentation
  - scripts for data and results visualization
- raw_data: links to our datasets

### Team Members and Task Breakdown
#### Huang Ziyu
- model training: vision transformer
- data collection
- data augmentation
- (mainly in charge) final presentation video edit
#### Li Zhaofeng
- model training: LSTM
- data analysis
- (mainly in charge) final report organization
#### Wang Yuchen
- model training: MLP, CNN
- data cleansing
- (mainly in charge) final presentation slides, github repo

### Project Introduction
COVID-19 has greatly affected people's lives. Masks have proved to be an effective method to stop the spread of the virus since 2020. Although Singapore has partially loosened its coronavirus controls, people are still required to wear masks in public places such as buses and subways. Therefore, intelligent identification of whether passengers are wearing masks is a research topic of great significance. 

We aim to locate human faces in the image, and detect whether the person is wearing a mask or not. The classification results are divided into three categories: face with a mask, face with no masks, and no human faces. The method is elaborated in the proposed solution part.

We expect our model to achieve considerable accuracy without loss of generalizability and interpretability. Through visualization and cross-model comparisons, we are able to provide both qualitative and quantitative performance analysis for our experimental results, which inspire future discussions into facial mask detection problem.

### Problem Formulation
Facial Mask Detection is a classification task where the input is an image, $i$, and the target is an integer indicator, $c$, of the predicted label $l$.

Formally, given a database $\mathcal{D}$, where each sample is an image-label pair $\langle i, c \rangle$, we aim to learn a predictor, $p$, that maximizes the prediction accuracy.

### Proposed Model Performance
acc = 98.5%
