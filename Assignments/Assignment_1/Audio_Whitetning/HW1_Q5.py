# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 18:35:01 2022

@author: Rohit
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import fft
import winsound

def read_data(var,playSound = 0):
    '''
    Reads the Audio file from the given path specified.
    
    Parameters
    ----------
    var : string
            name of audio file
    playSound : int, optional
            if 1 sound is played. The default is 0.

    Returns
    -------
    x : numpy 1D array
        The Amplitude of Audio data.
    sampleRate : int
        Sampling Rate of Audio File.

    '''
    data_dir="G:\\IISc\\MLSP\\Assignment\\HW-1 Q-5\\Data\\speechFiles\\"
    data_loc = data_dir+var
    sampleRate,x = wavfile.read(data_loc)
    # For Playing sound
    if playSound:
        winsound.PlaySound(data_loc, winsound.SND_FILENAME)  
    return x,sampleRate

def calculate_hamming_window(window_length):
    '''
    Computes Hamming Window of given window length.

    Parameters
    ----------
    window_length : int
        Length of Window.

    Returns
    -------
    hamm_win : 1D array
        hamming window of length = window_length.

    '''
    # x = np.asarray(x)
    hamm_win = np.asarray(np.hamming(window_length))
    return hamm_win

def apply_hamming_and_strip(x,hamm_win):
    '''
    Multiply elementwise the given signal x with hamming window.Both must be of same size = 400
    and then takes the middle 256 values from a 400 sized signal i.e. from index 72 - 328

    Parameters
    ----------
    x : 1D array
        input signal.
    hamm_win :  1D array
        hamming window of length 

    Returns
    -------
    vec_hamm_ : 1D array
        256 length signal multiplied with Hamming window.
    '''
    vec_hamm_ = x * hamm_win
    # Stripping feom middle of window to get 256 length array
    vec_hamm_ = vec_hamm_[72:328]
    # vec_hamm_ = vec_hamm_[:256]
    return vec_hamm_
    
def wav_to_spectogram(x,samp_rate,window,stride):
    '''
    Converts the input Audio File representes as 1D array to Spectogram
    by doing FFT transformation

    Parameters
    ----------
    x : 1D array
        input signal.
    samp_rate : int
        Sampling Rate: no of samples taken at each second.
    window : int(in ms)
        window length for one feature.
    stride : int(in ms)
        hop distance between two successive windows.

    Returns
    -------
    x_feature_vec : numpy ndarray
        a (NXD) array which represents the spectogram of the given audio file .
        where N = (x.shape[0] - window_length) / stride_length
              D = 128

    '''
    window_length = int(samp_rate * window * 0.001)
    stride_length = int(samp_rate * stride * 0.001)
    num_frames = 1 + int((x.shape[0] - window_length) / stride_length)
    x_feature_vec = []
    print(num_frames,window_length,samp_rate,stride_length,x.shape)
    hamm_win = calculate_hamming_window(window_length)
    for i in range(num_frames):
        single_vec = x[(i * stride_length) : (i * stride_length + window_length)]
        single_vec_hamm_ = apply_hamming_and_strip(single_vec, hamm_win)
        single_vec_to_fft = fft.rfft(single_vec_hamm_)
        # single_vec_to_fft = np.fft.fftshift(single_vec_to_fft)
        single_vec_to_fft = np.log(np.abs(single_vec_to_fft)+1)
        x_feature_vec.append(single_vec_to_fft)
    x_feature_vec = np.array(x_feature_vec)
    return x_feature_vec


def plot_histogram_(x1,x2,var1,var2,var = "",is_spec=False):
    '''
    plots distribution of data matrix x

    Parameters
    ----------
    x1,x2 : numpy array (1D or nD)
        input array whose distribution needs to be seen.
    var1,var2,var : string
        name of data files x1,x2 respectively and heading of plot.
    is_spec : bool (optional)
        is the data a spectogram . Defaults to False
    
    Returns
    -------
    None.

    '''
    if is_spec == True:
        var1 = var1 + " spectrogram"
        var2 = var2 + " spectrogram"
        
    plt.figure(figsize = (10,8))
    plt.suptitle("Distribution Plot " + var )
    
    plt.subplot(1,2,1)
    plt.title(var1 +" file")
    plt.xlabel("No of Samples")
    # plt.ylabel("Frequency")
    plt.hist(x1.ravel(),bins= 'auto')
    
    plt.subplot(1,2,2)
    plt.title(var2 +" file")
    plt.xlabel("No of Samples")
    # plt.ylabel("Frequency")
    plt.hist(x2.ravel(),bins= 'auto')
    
    plt.show()

def plot_(x1,x2,var1,var2):
    '''
    plots Amplitude vs Time plot of data matrix x

    Parameters
    ----------
    x1,x2 : numpy array 1D
        input array 
    var1,var2 : string
        name of data files x1,x2 respectively.

    Returns
    -------
    None.

    '''
    plt.figure(figsize = (10,8))
    plt.suptitle("Amplitude vs Time Plot" )
    plt.subplot(2,1,1)
    plt.title(var1 +" file")
    plt.xlabel("Time (in sec)")
    plt.ylabel("Amplitude")
    plt.plot(np.linspace(0,3.125,50000),x1)
    
    plt.subplot(2,1,2)
    plt.title(var2 +" file")
    plt.xlabel("Time (in sec)")
    plt.ylabel("Amplitude")
    plt.plot(np.linspace(0,3.125,50000),x2)
    plt.show()

def plot_spectogram_(x1,x2,var1,var2):
    '''
    plots spectogram of data matrix x

    Parameters
    ----------
    x1,x2 : numpy array 2D
        input array whose spectogram needs to be seen.
    var1,var2 : string
        name of data files x1,x2 respectively.

    Returns
    -------
    None.

    '''
    plt.figure(figsize = (10,8))
    plt.suptitle("Spectogram Plot")
    
    plt.subplot(2,1,1)
    plt.title(var1)
    plt.xlabel("No of Samples")
    plt.ylabel("Frequency")
    plt.imshow(x1,cmap='viridis')
    
    plt.subplot(2,1,2)
    plt.title(var2)
    plt.xlabel("No of Samples")
    plt.ylabel("Frequency")
    plt.imshow(x2,cmap = 'viridis')
    plt.show()

def reduce_dim_to_half(x):
    '''
    reduce dimension of each vector in x from (N x 1) to  ((N/2) x 1)

    Parameters
    ----------
    x : numpy 2D array
        DESCRIPTION. ---> Data Matrix

    Returns
    -------
    x : numpy 2D array
        DESCRIPTION. ---> reduced dim Data Matrix

    '''
    N = int(x.shape[1]/2)
    x = x[: , :N]
    return x


def compute_mean(x):
    '''
    Assumes DxN data matrix

    Parameters
    ----------
    x : numpy nd array 2D
        input array.

    Returns
    -------
    x_bar : Dx1 array
        average of input matrix x.

    '''
    x_bar = np.mean(x,axis = 1).reshape(-1,1)
    # x_bar = (np.ones(shape=(x.shape)) @ x)/x.shape[1]
    return x_bar


def normalize_data(x):
    '''
    Centers the data by removing mean

    Parameters
    ----------
    x : DxN numpy array
        input data matrix.

    Returns
    -------
    x_norm : DxN numpy array
        centered data matrix.

    '''
    x_mean = compute_mean(x)
    x_norm = x - x_mean
    return x_norm

def compute_covariance(x):
    '''
    Computes covariance matrix of array

    Parameters
    ----------
    x : DxN numpy array
        input data matrix.

    Returns
    -------
    cov_x : DxD numpy array
        covariance of input data matrix.

    '''
    x_norm = normalize_data(x)
    N = x.shape[1]
    cov_x = (x_norm @ x_norm.T) / N
    return cov_x
    
def whitening_transform(x):
    '''
    Computes the whitening transformation on the input data and returns its parameter W
    which does the transformation

    Parameters
    ----------
    x :  DxN numpy array
        input data matrix.

    Returns
    -------
    W : (DxD) numpy array
        2D matrix which does the whitening transformation.
    whitened_x : (NxD) numpy array
        whitened data matrix.

    '''
    x_norm = normalize_data(x)
    cov_x = compute_covariance(x)
    e_val,e_vec = np.linalg.eigh(cov_x)
    
    e_val = e_val[::-1]
    e_vec = e_vec[:,::-1]
    
    lamda_halfs = np.diag(1 / np.sqrt(e_val))
    U = e_vec
    
    W = lamda_halfs @ U.T
    
    whitened_x = W @ x_norm
    
    # print(x_T.shape , whitened_x.shape)
    
    # check if covariance of whitened data is indeed Identity
    # cov_w_x = compute_covariance(whitened_x)
    # print(np.allclose(np.diag(np.ones(cov_w_x.shape[1])),cov_w_x,atol = 1e-5))
    
    return W,whitened_x
    
    
def compute_average_non_diagonal_(cov1):
    '''
    Compute average of the absolute value of the non-diagonal entries of the sampled 
    covariance matrix of the “whitenned” data

    Parameters
    ----------
    cov1 : numpy ndarray
          (DxD) numpy array covariance matrix

    Returns
    -------
    average : flaot
            Returns average of non-daigonal entries

    '''
    av_cov = np.abs(cov1) 
    (N1,N2) = av_cov.shape
    sum_ = 0
    N = 0
    for i in range(N1):
        for j in range(N2):
            if i != j:
                sum_ += av_cov[i,j]
                N += 1
    average = sum_ / N
    # print(average,N,N1)
    return average


def part_A(x_clean_,x_noisy_):
    '''
    Assume that each speech frame (128 dimensional vector) is independent of each other.
    From the clean files, compute the whitening transform. Apply the transform on the
    noisy speech features. Find the average of the absolute value of the non-diagonal
    entries of the sampled covariance matrix of the “whitenned” clean and noisy speech
    features.
    
    Parameters
    ----------
    x_clean_ :  numpy ndarray
                (DxN) numpy array of clean.wav file
    x_noisy_ :  numpy ndarray
                (DxN) numpy array of noisy.wav file

    Returns
    -------
    None.

    '''
    
    '''Obtaining Whitening Parameter W from clean.wav'''
    W,whitened_x_clean = whitening_transform(x_clean_)
    # print(compute_covariance(whitened_x_clean))
    
    
    '''Performing Whitening operation on noisy.wav file with W parameter obtained from clean.wav '''
    whitened_noisy_x = W @ (x_noisy_ - compute_mean(x_clean_))
    # print("whitened_noisy_x.shape ",whitened_noisy_x.shape)
    
    plot_histogram_(x_clean_, x_noisy_, "Clean", "Noisy","Spectrogram Before Whitening")
    
    plot_histogram_(whitened_x_clean, whitened_noisy_x, "Clean", "Noisy","Spectrogram After Whitening on Clean Data")
    
    ''' sampled cov matrix of "whitened" clean and noisy speech feature '''
    whitened_cov_mat_clean = compute_covariance(whitened_x_clean)
    whitened_cov_mat_noisy = compute_covariance(whitened_noisy_x)
    
    # print(whitened_cov_mat_clean,whitened_cov_mat_noisy)
    
    '''Compute average of non-daiagonal of cov matrices BEFORE whitening'''
    avg1_ = compute_average_non_diagonal_(compute_covariance(x_clean_))
    avg2_ = compute_average_non_diagonal_(compute_covariance(x_noisy_))
     
    print("average of the absolute value of the non-diagonal entries of the ",
          "sampled covariance matrix BEFORE whitening \n","clean speech features is ",avg1_,
          "\n noisy speech features is ",avg2_,"-"*50)
    
   
    avg1_ = compute_average_non_diagonal_(whitened_cov_mat_clean)
    avg2_ = compute_average_non_diagonal_(whitened_cov_mat_noisy)
    
    print("PART-A average of the absolute value of the non-diagonal entries of the ",
          "sampled covariance matrix of the \n","“whitenned” clean speech features is ",avg1_,
          "\n “whitenned” noisy speech features is ",avg2_,"*"*50)
    # print("PART-0",np.var(x_clean_),np.var(x_noisy_),np.mean(x_clean_),np.mean(x_noisy_))
    # print("PART-A",np.var(whitened_x_clean),np.var(whitened_noisy_x),np.mean(whitened_x_clean),np.mean(whitened_noisy_x))
    
    ##############################################################################3
    
def part_B(x_clean_,x_noisy_):
    '''
    Repeat the procedure in part A by reversing the roles of clean and noisy files.

    Parameters
    ----------
    x_clean_ :  numpy ndarray
                (DxN) numpy array of clean.wav file
    x_noisy_ :  numpy ndarray
                (DxN) numpy array of noisy.wav file

    Returns
    -------
    None.

    '''
    
    '''Obtaining Whitening Parameter W from noisy.wav'''
    W,whitened_x_noisy = whitening_transform(x_noisy_)
    # print(compute_covariance(whitened_x_clean))
    
    
    '''Performing Whitening operation on clean.wav file with W parameter obtained from noisy.wav '''
    whitened_clean_x = W @ (x_clean_ - compute_mean(x_noisy_))
    # print("whitened_noisy_x.shape ",whitened_clean_x.shape)
    
    
    ''' sampled cov matrix of "whitened" clean and noisy speech feature '''
    whitened_cov_mat_noisy = compute_covariance(whitened_x_noisy)
    whitened_cov_mat_clean = compute_covariance(whitened_clean_x)
    # print(whitened_cov_mat_clean,whitened_cov_mat_noisy)
    
    plot_histogram_(whitened_clean_x, whitened_x_noisy, "Clean", "Noisy","Spectrogram After Whitening on Noisy Data")
    
    avg1_ = compute_average_non_diagonal_(whitened_cov_mat_clean)
    avg2_ = compute_average_non_diagonal_(whitened_cov_mat_noisy)
    
    print("PART-B average of the absolute value of the non-diagonal entries of the ",
          "sampled covariance matrix of the \n","“whitenned” clean speech features is ",avg1_,
          "\n “whitenned” noisy speech features is ",avg2_,"*"*50)
    # print("PART-B",np.var(whitened_clean_x),np.var(whitened_x_noisy),np.mean(whitened_clean_x),np.mean(whitened_x_noisy))
    
    ##############################################################################33333333333    


if __name__ == "__main__":
    
    '''Reading clean.wav and noisy.wav file'''
    x_clean,sampRate = read_data("clean.wav",playSound=0)
    x_noisy,sampRate = read_data("noisy.wav",playSound=0)
    
    '''Converting read 1D speech array to (NxD) spectogram file'''
    x_clean_spectrogram = wav_to_spectogram(x_clean, sampRate, 25, 10)
    # print(x_clean_spectrogram.shape,x_clean_spectrogram[1,1] ,x_clean_spectrogram[1,-1])
    x_noisy_spectogram = wav_to_spectogram(x_noisy, sampRate, 25, 10)
    print(x_clean_spectrogram.shape)
    '''Taking only first 128 dimensions'''
    # x_clean_spectrogram = reduce_dim_to_half(x_clean_spectrogram)
    # x_noisy_spectogram = reduce_dim_to_half(x_noisy_spectogram)
    # print(x_clean_spectrogram.shape,x_noisy_spectogram.shape)
    
    '''Converting data matrices to (DxN) array'''
    x_clean_spectrogram = x_clean_spectrogram.T
    x_noisy_spectogram = x_noisy_spectogram.T
    
    '''Plot of Amplitude vs time , distribution of data(histogram) , spectrogram plot of Audio files
    side by side'''
    var1 ="Clean.wav"
    var2 = "Noisy.wav"
    plot_(x_clean,x_noisy,var1,var2)
    plot_histogram_(x_clean,x_noisy,var1,var2)
    plot_histogram_(x_clean_spectrogram,x_noisy_spectogram,var1,var2,is_spec=True)
    plot_spectogram_(x_clean_spectrogram,x_noisy_spectogram,var1,var2)
    
    '''From the clean files, compute the whitening transform. Apply the transform on the 
    noisy speech features. Find the average of the absolute value of the non-diagonal
    entries of the sampled covariance matrix of the “whitenned” clean and noisy speech
    features.'''
    part_A(x_clean_spectrogram,x_noisy_spectogram)
    
    
    '''From the noisy files, compute the whitening transform. Apply the transform on the 
    clean speech features. Find the average of the absolute value of the non-diagonal
    entries of the sampled covariance matrix of the “whitenned” clean and noisy speech
    features.'''
    part_B(x_clean_spectrogram,x_noisy_spectogram)

