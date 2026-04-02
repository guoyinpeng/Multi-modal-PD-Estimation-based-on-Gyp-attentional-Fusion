import os
import pandas as pd
import numpy as np
import librosa
import librosa.display

import seaborn as sns
import matplotlib.pyplot as plt
from skimage import io, transform
from mpl_toolkits.mplot3d import Axes3D


def cosine_similarity(u, v):
    '''Calculate the similarity between 1D arrays'''
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def similarity_matrix(array):
    '''Calculate the similarity matrix by given a 2D array'''
    shape = array.shape
    similarity = np.zeros((shape[0], shape[0]))

    for i in range(shape[0]):
        for k in range(shape[0]):
            similarity[i][k] = cosine_similarity(array[i], array[k])

    return similarity


########################################################################################################################
# NOTE：绘图相关
########################################################################################################################

def show_spectrogram(audio_feature, audio_parameters, y_axis="log"):
    """Show log-spectrogram for a batch of samples.
    Arguments:
        audio_feature: 2D numpy.ndarray, extracted audio feature (spectra) in dB
        audio_parameters: dict, all parameters setting of STFT
                          we used for feature extraction
        y_axis: certain string, scale of the y axis. could be 'linear' or 'log'
    Return:
        plot the spectrogram
    """

    # transpose, so the column corresponds to time series
    audio_feature = np.transpose(audio_feature)

    plt.figure(figsize=(25, 10))
    im = librosa.display.specshow(audio_feature,
                                  sr=audio_parameters['sample_rate'],
                                  hop_length=audio_parameters['hop_size'],
                                  x_axis="time",
                                  y_axis=y_axis)
    plt.colorbar(format="%+2.f dB")
    return im


def show_mel_filter_banks(filter_banks, audio_parameters):
    """Show Mel filter bank for a batch of samples.
    Arguments:
        filter_banks: 2D numpy.ndarray, please use self.filter_banks to get the value,
                      but make sure load_audio(spectro_type='mel_spectrogram') is called
        audio_parameters: dict, all parameters setting of STFT
                                we used for feature extraction
    Return:
        visualize the mel filter banks
    """
    plt.figure(figsize=(25, 10))
    im = librosa.display.specshow(filter_banks,
                                  sr=audio_parameters['sample_rate'],
                                  x_axis="linear")
    plt.colorbar(format="%+2.f")
    return im


def show_similarity_matrix(text_feature, start_sent, sent_len):
    '''plot the result of similarity matrix as heatmap'''
    # calculate similarity
    similarity = similarity_matrix(text_feature['sentence_embeddings'][int(start_sent):int(start_sent + sent_len)])
    # plot heatmap
    plt.figure(figsize=(16, 16))
    heatmap = sns.heatmap(similarity, annot=True, fmt='.2g')  # cbar_kws={'label': 'correlation'}
    # set scale label
    heatmap.set_xticklabels(text_feature['indices'][int(start_sent):int(start_sent + sent_len)])  # rotation=-30
    heatmap.set_yticklabels(text_feature['indices'][int(start_sent):int(start_sent + sent_len)], rotation=0)
    # set label
    plt.xlabel("sentence number in conversation")
    plt.ylabel("sentence number in conversation")
    plt.show()