# Image-Caption-Generator
Generating Image Captions using Long-Short Term Memory and Recurrent Neural Network.

## Introduction 
- Image Caption Generation is outputting a correct and concise description of a given image.
- Automated captioning of images demands an understanding of the fields of Computer Vision as well as Natural Language Processing.
- Automated captioning is useful in situations where an image takes time to load or is unavailable. 
- These captions can also be used to classify the images.

## Baseline
- We are using the paper “Learning CNN-LSTM Architectures for
Image Caption Generation” by Moses Soh as our baseline.
- This paper has proposed using a deep CNN to generate a vectorized
representation of an image that is given as input to the LSTM to
generate the caption. This is the top-down approach.
- The main drawback of captions generated by models based on the
top-down approach is their inability to capture minute details.
- We have implemented a model that uses a RNN in order to overcome this drawback.

## Dataset
- Flickr8k dataset contains 8092 photographs along with text descriptions by their photographs
- The dataset is about 8 GB in size
- Extract features from images using VGG16 
- Load descriptions for each image
- Clean the descriptions by removing 
punctuations, converting all words to 
lowercase and removing numbers 

## Model Architecture

- Photo Feature Extractor 
  - input photo features as vector of 4096 elements
  - 50% dropout
- Sequence Processor
  - input sequences of 34 words
  - Embedding layer that used to ignore padding
  - 50% dropout 
  - LSTM with 256 memory units
- Decoder
  - Add both outputs from the two units
  - Dense layer to make prediction of next word

