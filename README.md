# facial_mask_detection

This project serves as the final project for CS5242: Neural Networks and Deep Learning in AY2022/23, Semester 1.

### Introduction
COVID-19 has greatly affected people's lives. Masks have proved to be an effective method to stop the spread of the virus since 2020. Although Singapore has partially loosened its coronavirus controls, people are still required to wear masks in public places such as buses and subways. Therefore, intelligent identification of whether passengers are wearing masks is a research topic of great significance. 

### Description
We aim to locate human faces in the image, and detect whether the person is wearing a mask or not. The classification results are divided into three categories: face with a mask, face with no masks, and no human faces. The method is elaborated in the proposed solution part.

We expect our model to achieve considerable accuracy without loss of generalizability and interpretability. Through visualization and cross-model comparisons, we are able to provide both qualitative and quantitative performance analysis for our experimental results, which inspire future discussions into facial mask detection problem.

### Data Collection
In order to ensure the randomness and representatives of the dataset, we decided to obtain pictures of wearing masks from different websites using scrapper, including but not limited to Flickr, Google, and Video Snapshot. Use the keyword 'people', 'people wearing mask' and 'bus' to collect three types of picture, so that there is no need to manually make labels. To make the follow-up task more accurate, we used YOLO to capture faces. Then to ensure the accuracy of the dataset, the images will be manually checked to guarantee if the face was captured correctly, or if the 'mask' label was correct.

### Problem Formulation
Facial Mask Detection is a classification task where the input is an image, $i$, and the target is an integer indicator, $c$, of the predicted label $l$.

Formally, given a database $\mathcal{D}$, where each sample is an image-label pair $\langle i, c \rangle$, we aim to learn a predictor, $p$, that maximizes the prediction accuracy.

### Author
- Huang Ziyu
- Li Zhaofeng
- Wang Yuchen
