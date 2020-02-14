# 380-music-representation
## Introduction
This repository contains the code used for my undergradute computer science project (COMPSCI-380) which was part of my studies at the University of Auckland.

In this project, I look at two ways of representing piano music from the deep learning literature, train models and generate sequences using each of these representations, and experiment with the introduction of chroma, a succinct musical descriptor that could possibly help a model better abstract to higher level features.

The dataset used is V2.0.0 of [the MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro), a collection of more than 1000 performances and 6 million notes from the International Piano-e-Competition.

Code is written in python, and Keras and TensorFlow are used for building and training models. The [pretty-midi](https://github.com/craffel/pretty-midi) library is used for reading and writing midi files. 

## Notebooks and generating new sequences
Three notebooks are provided, two for generating sequences using each of the representations, and one for data exploration.

## Models
Weights from various training runs are provided. In later experiments I was keeping much better track of the parameters used by automatically generating a text file describing that particular training run. 