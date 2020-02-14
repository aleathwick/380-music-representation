# 380-music-representation
## Introduction
This repository contains the code used for my undergradute computer science project (COMPSCI-380) which was part of my studies at the University of Auckland.

In this project, I look at two ways of representing piano music from the deep learning literature, train models and generate sequences using each of these representations, and experiment with the introduction of chroma, a succinct musical descriptor that could possibly help a model better abstract to higher level features.

The dataset used is V2.0.0 of [the MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro), a collection of more than 1000 performances and 6 million notes from the International Piano-e-Competition.

Code is written in python, and Keras and TensorFlow are used for building and training models. The [pretty-midi](https://github.com/craffel/pretty-midi) library is used for reading and writing midi files.

A note on naming confusion: The two representations used are *NoteTuple* (NT) and *Performance Representation* (PF) -- these are the names used in the literature. But early on in the project, I was calling them *Note Bin* and *Oore*, and as a result these are the two names used throughout my code.


The original papers:

PF is first introduced in depth in [*This time with feeling: Learning expressive musical performance*](https://arxiv.org/abs/1808.03715). There is an earlier paper , but it is brief.

NT is introduced in [*Transformer-NADE for Piano Performances*](https://nips2018creativity.github.io/doc/Transformer_NADE.pdf)

## Notebooks & Scripts
### Sequence Generatioin
Two notebooks for generating new sequences are provided, one for each of the representations. Some ready-trained weights are provided for use with the sequence generation notebooks. For some examples of generated sequences, check out [this youtube playlist](https://www.youtube.com/playlist?list=PLCO5IgjyszQvCVXG4f_JiPaQwcCvoQCpN).

### Data Making & Analysing
A notebook is provided for creating data in each representation, starting from the raw midi files of the MAESTRO dataset.
Also provided is the notebook I used to plot and compare models, and explore the data.

### Model Training
Four scripts are provided for training models: one each for PF and NT with and without chroma. These scripts generate loss plots, loss history, a log file describing model and data configuration, and model weights that can be loaded later for prediction.

## A brief summary of my work
In short, I found that NT is much easier to generate human like sequences with. A lot of that probably comes down to the primitive sampling method I used. PR splits up attributes of a note into separate events, whereas NT has the model predict all attributes of a note in a single timestep, thus introducing some joint modelling of probability that PR would need beam search to emulate. The inventors of PR used beam search, I did not.

Chroma (12 indicator variables indicating presence or absence of a pitch class at a given time step) worked unexpectedly well in bringing down validation loss. I expected a modified version of chroma, in which the indicator variables are replaced with continuous values so that stronger signal can be given for lower notes, would work better. There can be an awful lot of high notes caught in the pedal in piano music, which can flood the chroma, making it less useful - or so I thought. I tried normal chroma, a weighted version of chroma, and a version with only the pitch class of the lowest note. All three caused training loss to drop faster when compared to replacing these features with zeros, but only normal chroma had a significant impact on validation loss. *How* it achieves this impact on validation loss is left to future work. Probably, it is either providing useful higher level harmonic signal, or it is acting as memory by providing a picture of the current harmonic/sonic environment (giving the model ears, so to speak - the pianist can hear what is happening currently, and doesn't have to remember all notes played in the past, so why should a model?).

For more information, you are welcome to read the full report.

## Code
The following are found in the modules folder:
- midiMethods: Code for processing pretty midi (pm) objects, converting between pm and each of the two representations and back, and generating different kinds of chroma for each of the representations.
- dataMethods: Code for working with data on a larger scale, e.g. converting files from a list of filenames into training examples.
- mlClasses: Contains data generators for each representation. Of much use in figuring these out was [this blog post](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly) by Afshine Amidi and Shervine Amidi.
- models: Contains the various models and code for generating sequences from models. The sequence generation code, and the method of sampling from the output distribution with the softmax temperature optimized to balance between craziness and conservativeness of sequences, is inspired by a very good [tutorial on text generation in tensorflow](https://www.tensorflow.org/tutorials/text/text_generation#the_prediction_loop).

Finally, there is a folder called obsolete, which contains code from when I first played around with performance representation, a while before undertaking this project. At the time, I had a poor understanding of how to use keras properly, I didn't understand it was necessary to sample from the output distribution or use beam search, and I didn't have access to a GPU. But though no sequences were successfuly generated, I learnt a huge amount.

Modules are not named according to normal python conventions, but use camelCase. Apologies. 