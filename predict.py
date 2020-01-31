import layers
import cv2
import numpy as np
import utils
from PIL import Image
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from computationalgraph import ComputationalGraph

def load_image(test_dir, IMG_SIZE=227):
    for r, d, f in os.walk(test_dir):
        for filename in f:
            #X shape is [WIDTH, HEIGHT, CHANNELS]
            X = cv2.imread(test_dir + "/" + filename)
            #Resize X to [IMG_SIZE, WIDTH, HEIGHT]
            X = cv2.resize(X, (IMG_SIZE, IMG_SIZE))
            #Transform X to shape [CHANNELS, HEIGHT, WIDTH]
            X = X.transpose(2, 0, 1)
            #X = (X/255)*0.99 + 0.01
            X = X[np.newaxis, :, :, :]
            X = np.asarray(X, dtype=float)

            return X

def get_mean_image(labels, IMG_SIZE=227):
    images = []
    n_labels = len(labels)

    print()
    print("[+] loading images...")
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


test_dir = "test_images"
labels = ["bart_simpson", "homer_simpson", "marge_simpson", "lisa_simpson"]

mean_image = get_mean_image(labels)



X = load_image(test_dir, IMG_SIZE=227)

X -= mean_image

n_classes = len(labels)

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
print()

Y = graph.predict(X)

print()
print("Y:")
print(Y)

max_index = np.argmax(Y)

print()
print("[+]", labels[max_index] + ":", np.round((Y[max_index][0]*100),2), "%")
