## Multimodal Detection of Sensitive Content

This repository contains the implementation of our paper "Empowering First Responders through Automated Multimodal Content Moderation" (accepted at IEEE ICCC 2018). The paper is available here: http://precog.iiitd.edu.in/pubs/empowering-first-responders.pdf

## Abstract

Social media enables users to spread information and opinions, including in times of crisis events such as riots, protests or uprisings. Sensitive event-related content can lead to repercussions in the real world. Therefore it is crucial for first responders, such as law enforcement agencies, to have ready access, and the ability to monitor the propagation of such content. Obstacles to easy access include a lack of automatic moderation tools targeted for first responders. Efforts are further complicated by the multimodal nature of content which may have either textual and pictorial aspects. In this work, as a means of providing intelligence to first responders, we investigate automatic moderation of sensitive event-related content across the two modalities by exploiting recent advances in Deep Neural Networks (DNN). We use a combination of image classification with Convolutional Neural Networks (CNN) and text classification with Recurrent Neural Networks (RNN). Our multilevel content classifier is obtained by fusing the image classifier and the text classifier. We utilize feature engineering for preprocessing but bypass it during classification due to our use of DNNs while achieving coverage by leveraging community guidelines. Our approach maintains a low false positive rate and high precision by learning from a weakly labeled dataset and then, by learning from an expert annotated dataset. We evaluate our system both quantitatively and qualitatively to gain a deeper understanding of its functioning. Finally, we benchmark our technique with current approaches to combating sensitive content and find that our system outperforms by 16% in accuracy.

## Dataset

Contact guptadivam@gmail.com for the dataset.

## Requirements

Dependencies can be installed using requirements.txt file by running:

```
$ pip install -r requirements.txt
```

## Instructions to run

To pre-train the Image Model:
```
cd train
python trainCNN.py
```

To train the LSTM model on the weekly annotated dataset: 
```
cd train
python trainLSTM_weaklyLabeled.py
```

To train the multimodal classiifer: 
```
cd train
python trainMultiModel_LSTM_CNN.py
```

For training we assume a list of dictionaries for each datapoint. In each dictionary the key 'img' contains the image path, the key 'text' contains the text of the tweet and the key 'label' contains a binary 0/1 label.

To cite, please use the following:

```
@inproceedings{gupta2018empowering,
  title={Empowering First Responders through Automated Multimodal Content Moderation},
  author={Gupta, Divam and Sen, Indira and Sachdeva, Niharika and Kumaraguru, Ponnurangam and Balaji, Arun Buduru},
  booktitle={2018 IEEE International Conference on Cognitive Computing (ICCC)}, 
  year={2018}
}
```
