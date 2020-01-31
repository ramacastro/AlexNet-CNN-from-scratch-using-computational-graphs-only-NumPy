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
                targets.append(T)
                
    images = np.stack(images, axis=0)
    images -= images.mean(axis=0)

    return images, targets


#labels = ["billie", "car", "elon", "earth", "dog"]
labels = ["bart_simpson", "homer_simpson", "marge_simpson", "lisa_simpson"]

images, targets = load_images(labels, IMG_SIZE=227)

epochs = 2

n_classes = len(labels)
n_examples = len(images)

cost_values = []

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


try:
    print()
    print("[+] N_EXAMPLES:", n_examples)
    for e in range(epochs):
        images_shuffled, targets_shuffled = shuffle(images, targets)
        total_predicted_ok = 0
        total_cost = 0
        print("\n------------------[EPOCH " + str(e+1) + "]------------------\n")
        for i in tqdm(range(n_examples)):
            X = images_shuffled[i]
            T = targets_shuffled[i]

            Yi, loss_value, predicted_ok = graph.train(X, T)
            
            total_cost += loss_value
            total_predicted_ok += predicted_ok

        total_cost /= n_examples

        cost_values.append(total_cost)
        accuracy = np.round((total_predicted_ok/n_examples)*100, 2)

        print("\n[+] COST:", loss_value)
        print("[+] PREDICTED_OK:", total_predicted_ok)
        print("[+] ACCURACY:", accuracy, "%")

except KeyboardInterrupt:
    pass

print("\n---------------------------------------------\n")

graph.save_parameters()

plt.plot(cost_values)
plt.show()















'''
network_name = "RamaNet"

graph = ComputationalGraph(network_name, loss_function="cross-entropy", lr=1e-2, reg=1e-5)

graph.add_layer(layers.Convolution("conv1", kernel_size=3, kernel_channels=3, n_kernels=16, stride=1, same_padding=True, logging=False))
graph.add_layer(layers.Activation(activation="relu", logging=False))
graph.add_layer(layers.Convolution("conv2", kernel_size=3, kernel_channels=16, n_kernels=16, stride=1, same_padding=True, logging=False))
graph.add_layer(layers.Activation(activation="relu", logging=False))
graph.add_layer(layers.MaxPooling(window_size=2, window_channels=16, stride=2, logging=False))

graph.add_layer(layers.Convolution("conv3", kernel_size=3, kernel_channels=16, n_kernels=32, stride=1, same_padding=True, logging=False))
graph.add_layer(layers.Activation(activation="relu", logging=False))
graph.add_layer(layers.Convolution("conv4", kernel_size=3, kernel_channels=32, n_kernels=32, stride=1, same_padding=True, logging=False))
graph.add_layer(layers.Activation(activation="relu", logging=False))
graph.add_layer(layers.MaxPooling(window_size=2, window_channels=32, stride=2, logging=False))

graph.add_layer(layers.Convolution("conv5", kernel_size=3, kernel_channels=32, n_kernels=64, stride=1, same_padding=True, logging=False))
graph.add_layer(layers.Activation(activation="relu", logging=False))
graph.add_layer(layers.Convolution("conv6", kernel_size=3, kernel_channels=64, n_kernels=64, stride=1, same_padding=True, logging=False))
graph.add_layer(layers.Activation(activation="relu", logging=False))
graph.add_layer(layers.MaxPooling(window_size=2, window_channels=64, stride=2, logging=False))

graph.add_layer(layers.Convolution("conv7", kernel_size=3, kernel_channels=64, n_kernels=128, stride=1, same_padding=True, logging=False))
graph.add_layer(layers.Activation(activation="relu", logging=False))
graph.add_layer(layers.MaxPooling(window_size=2, window_channels=128, stride=2, logging=False))

graph.add_layer(layers.Convolution("conv8", kernel_size=3, kernel_channels=128, n_kernels=256, stride=1, same_padding=True, logging=False))
graph.add_layer(layers.Activation(activation="relu", logging=False))
graph.add_layer(layers.MaxPooling(window_size=2, window_channels=256, stride=2, logging=False))


graph.add_layer(layers.Flatten((4096, 1), logging=False))
graph.add_layer(layers.FullyConnected("fc1", 4096, 4096, activation="relu", logging=False))
graph.add_layer(layers.FullyConnected("fc2", 4096, 4096, activation="relu", logging=False))
graph.add_layer(layers.FullyConnected("fc4", 4096, n_classes, activation="softmax", logging=False))
'''