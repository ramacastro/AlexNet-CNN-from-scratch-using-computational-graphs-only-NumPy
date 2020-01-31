import layers
import cv2
import numpy as np
import utils
from PIL import Image
import computationalgraph


def load_images(filenames, IMG_SIZE=50):
    images = []

    for filename in filenames:
        #X shape is [WIDTH, HEIGHT, CHANNELS]
        X = cv2.imread(filename)
        #Resize X to [IMG_SIZE, WIDTH, HEIGHT]
        X = cv2.resize(X, (IMG_SIZE, IMG_SIZE))
        #Transform X to shape [CHANNELS, HEIGHT, WIDTH]
        X = X.transpose(2, 0, 1)
        #X = X[np.newaxis, :, :, :]
        images.append(X)

    return np.stack(images, axis=0)


X = load_images(["billie.png"], IMG_SIZE=50)
print(X.shape)

T = np.array([1,0,0,0,0])
T = T.T

######################################################################################################################
conv = layers.Convolution(kernel_size=3, kernel_channels=3, n_kernels=10, stride=1, same_padding=True)
X  = conv.forward(X)
print("[+] CONVOLVED_shape:", X.shape)
print()

activation = layers.Activation(activation="relu")
X = activation.forward(X)
print("[+] ACTIVATION_shape:", X.shape)
dActivation = activation.backward(np.ones_like(X))
print("dActivation_SHAPE:", dActivation.shape)
print()


maxpool = layers.MaxPooling(window_size=2, window_channels=10, stride=2)
X = maxpool.forward(X)
print("[+] MAXPOOLED_shape:", X.shape)
dMaxpool = maxpool.backward(np.ones_like(X))
print("dMaxpool_SHAPE:", dMaxpool.shape)

N, C, H, W = X.shape

flatten = layers.Flatten((C*H*W, N))
X = flatten.forward(X)
print(X.shape)

I, N = X.shape

fc1 = layers.FullyConnected("fc1", I, 1000, activation="relu")
X = fc1.forward(X)
print(X.shape)

fc2 = layers.FullyConnected("fc2", 1000, 10, activation="softmax")
Y = fc2.forward(X)
print(Y.shape)

cross_entropy = layers.CrossEntropy()
loss = cross_entropy.forward(Y, T)
print(loss)
######################################################################################################################

'''
print()

print("XSHAPE:", X.shape)
print()

N, C, H, W = X.shape

for n in range(N):
    for i in range(C):
        Xi = X[n][i]
        img = Image.fromarray(Xi)
        img = img.convert("LA")
        img_name = "output_example" + str(n) + "_filter" + str(i) + ".png"
        img.save(img_name)
        print("[+] " + img_name + " saved successfully")
'''