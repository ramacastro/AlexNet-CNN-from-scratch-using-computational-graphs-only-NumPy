import layers
import cv2
import numpy as np
import os
import utils
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from computationalgraph import ComputationalGraph
from sklearn.utils import shuffle

def get_mean_image(labels, IMG_SIZE=227):
    images = []
    n_labels = len(labels)

    print()
    print("[+] computing mean image...")
    print()

    for label in tqdm(labels):
        T = np.zeros((n_labels, 1))
        T[labels.index(label)] = 1
        
        for r, d, f in os.walk(label):
            for filename in f:
                path_to_file = label + "/" + filename
                #X shape is [WIDTH, HEIGHT, CHANNELS]
                X = cv2.imread(path_to_file)
                #Resize X to [IMG_SIZE, WIDTH, HEIGHT]
                X = cv2.resize(X, (IMG_SIZE, IMG_SIZE))
                #Transform X to shape [CHANNELS, HEIGHT, WIDTH]
                X = X.transpose(2, 0, 1)
                #X = (X/255)*0.99 + 0.01
                X = X[np.newaxis, :, :, :]
                X = np.asarray(X, dtype=float)

                images.append(X)
    
    images = np.stack(images, axis=0)
    print()
    return images.mean(axis=0)

def load_images(labels, IMG_SIZE=227):
    images = []
    targets = []
    n_labels = len(labels)

    print()
    print("[+] loading images...")
    print()

    for label in tqdm(labels):
        T = np.zeros((n_labels, 1))
        T[labels.index(label)] = 1
        
        for r, d, f in os.walk(label + "_test"):
            for filename in f:
                path_to_file = label + "_test/" + filename
                #X shape is [WIDTH, HEIGHT, CHANNELS]
                X = cv2.imread(path_to_file)
                #Resize X to [IMG_SIZE, WIDTH, HEIGHT]
                X = cv2.resize(X, (IMG_SIZE, IMG_SIZE))
                #Transform X to shape [CHANNELS, HEIGHT, WIDTH]
                X = X.transpose(2, 0, 1)
                #X = (X/255)*0.99 + 0.01
                X = X[np.newaxis, :, :, :]
                X = np.asarray(X, dtype=float)

                images.append(X)
                targets.append(T)
                
    mean_image = get_mean_image(labels, IMG_SIZE)
    
    images = np.stack(images, axis=0)
    images -= images.mean(axis=0)

    return images, targets


#labels = ["billie", "car", "elon", "earth", "dog"]
labels = ["bart_simpson", "homer_simpson", "marge_simpson", "lisa_simpson"]

images, targets = load_images(labels, IMG_SIZE=227)

n_classes = len(labels)
n_examples = len(images)


network_name = "AlexNet"

graph = ComputationalGraph(network_name, loss_function="cross-entropy", lr=1e-4, reg=1e-5)

graph.add_layer(layers.Convolution("conv1", kernel_size=11, kernel_channels=3, n_kernels=96, stride=4, same_padding=False, logging=False))
graph.add_layer(layers.Activation(activation="relu", logging=False))
graph.add_layer(layers.MaxPooling(window_size=3, window_channels=96, stride=2, logging=False))

graph.add_layer(layers.Convolution("conv2", kernel_size=5, kernel_channels=96, n_kernels=256, stride=1, same_padding=True, logging=False))
graph.add_layer(layers.Activation(activation="relu", logging=False))
graph.add_layer(layers.MaxPooling(window_size=3, window_channels=256, stride=2, logging=False))

graph.add_layer(layers.Convolution("conv3", kernel_size=3, kernel_channels=256, n_kernels=384, stride=1, same_padding=True, logging=False))
graph.add_layer(layers.Activation(activation="relu", logging=False))

graph.add_layer(layers.Convolution("conv4", kernel_size=3, kernel_channels=384, n_kernels=384, stride=1, same_padding=True, logging=False))
graph.add_layer(layers.Activation(activation="relu", logging=False))

graph.add_layer(layers.Convolution("conv5", kernel_size=3, kernel_channels=384, n_kernels=256, stride=1, same_padding=True, logging=False))
graph.add_layer(layers.Activation(activation="relu", logging=False))

graph.add_layer(layers.MaxPooling(window_size=3, window_channels=256, stride=2, logging=False))

graph.add_layer(layers.Flatten((9216, 1), logging=False))
graph.add_layer(layers.FullyConnected("fc1", 9216, 4096, activation="relu", logging=False))
graph.add_layer(layers.FullyConnected("fc2", 4096, 4096, activation="relu", logging=False))
graph.add_layer(layers.FullyConnected("fc3", 4096, n_classes, activation="softmax", logging=False))

graph.load_parameters()

try:
    total_predicted_ok = 0
    images_shuffled, targets_shuffled = shuffle(images, targets)
    print()
    print("[+] N_EXAMPLES:", n_examples)
    print()
    for i in tqdm(range(n_examples)):
        X = images_shuffled[i]
        T = targets_shuffled[i]

        Yi = graph.predict(X)

        predicted_ok = np.sum(np.argmax(Yi, axis=0) == np.argmax(T, axis=0))
        
        total_predicted_ok += predicted_ok


    accuracy = np.round((total_predicted_ok/n_examples)*100, 2)

    print()
    print("[+] PREDICTED_OK:", total_predicted_ok)
    print("[+] ACCURACY:", accuracy, "%")

except KeyboardInterrupt:
    pass

