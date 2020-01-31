import numpy as np
import gates
import utils
import standfordutils

class FullyConnected():
    def __init__(self, name, input_size, output_size, activation="relu", logging=False):
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.W = np.random.randn(output_size, input_size)*np.sqrt(2.0/input_size)
        self.B = np.zeros((output_size,1))
        self.regularize = False
        self.layer_type = "FullyConnected"
        self.trainable = True
        self.logging = logging

    def add_regularization(self, reg=1e-4):
        self.regularize = True
        self.reg = reg

    def regularization_loss(self):
        return np.sum(np.square(self.W))

    def forward(self, X):
        self.S = gates.MatMultGate()
        self.Z = gates.MatAddGate()

        if self.activation == "relu":
            self.A = gates.MatReLUGate()
        elif self.activation == "leaky_relu":
            self.A = gates.MatLeakyReLUGate()
        elif self.activation == "sigmoid":
            self.A = gates.MatSigmoidGate()
        elif self.activation == "softmax":
            self.A = gates.SoftmaxGate()

        S_out = self.S.forward(self.W, X)
        Z_out = self.Z.forward(S_out, self.B)
        A_out = self.A.forward(Z_out)

        if self.logging:
            print("[+] FULLYCONNECTED_OUT:", A_out.shape)

        return A_out

    def backward(self, dZ):
        dA = self.A.backward(dZ)
        dZ_S, self.dZ_B = self.Z.backward(dA)
        self.dW, dX = self.S.backward(dZ_S)

        if self.regularize:
            self.dW += self.reg*self.W

        return dX

    def update_parameters(self, lr):
        self.W -= lr*self.dW
        self.B -= lr*self.dZ_B

    def get_parameters(self):
        return self.W, self.B

    def set_parameters(self, W, B):
        self.W = W
        self.B = B


class Convolution():
    def __init__(self, name, kernel_size=3, kernel_channels=3, n_kernels=1, stride=1, same_padding=False, logging=False):
        self.name = name
        self.kernel_size = kernel_size
        self.kernel_channels = kernel_channels
        self.n_kernels = n_kernels
        self.stride = stride
        self.same_padding = same_padding
        self.W = 0.01*np.random.randn(n_kernels, kernel_size*kernel_size*kernel_channels)
        self.B = np.zeros((n_kernels, 1, 1))
        self.layer_type = "Convolution"
        self.trainable = True
        self.logging = logging
    
    def forward(self, X):
        self.N, C, X_n, _ = X.shape
        self.X_shape = X.shape

        if self.same_padding:
            self.padding = utils.get_same_padding(X_n, self.kernel_size, self.stride)
        else:
            self.padding = 0

        _, self.new_X_n, Xcol = standfordutils.im2col(X, self.kernel_size, self.kernel_size, self.padding, self.stride)
        
        self.S = gates.ConvMultGate()
        self.Z = gates.ConvAddGate()
        S_out = self.S.forward(self.W, Xcol)
        Z_out = self.Z.forward(S_out, self.B)
        Z_out = Z_out.reshape(self.N, self.n_kernels, self.new_X_n, self.new_X_n)
        
        if self.logging:
            print("[+] CONVOLUTION_OUT:", Z_out.shape)

        return Z_out
    
    def backward(self, dZ):
        dZ = dZ.reshape(self.N, self.n_kernels, self.new_X_n*self.new_X_n)
        dZ_S, self.dZ_B = self.Z.backward(dZ)
        self.dW, dXcol = self.S.backward(dZ_S)
        dX = standfordutils.col2im_indices(dXcol, self.X_shape, self.kernel_size, self.kernel_size, self.padding, self.stride)
        return dX

    def update_parameters(self, lr):
        self.W -= lr*self.dW
        self.B -= lr*self.dZ_B

    def get_parameters(self):
        return self.W, self.B

    def set_parameters(self, W, B):
        self.W = W
        self.B = B


class MaxPooling():
    def __init__(self, window_size=2, window_channels=3, stride=1, logging=False):
        self.window_size = window_size
        self.window_channels = window_channels
        self.stride = stride
        self.layer_type = "MaxPooling"
        self.trainable = False
        self.logging = logging

    def forward(self, X):
        self.S = gates.MaxPoolingGate()
        S_out = self.S.forward(X, self.window_size, self.stride)

        if self.logging:
            print("[+] MAXPOOLING_OUT:", S_out.shape)
            print()

        return S_out
    
    def backward(self, dZ):
        dS = self.S.backward(dZ)
        return dS



class Activation():
    def __init__(self, activation="relu", logging=False):
        self.activation = activation
        self.layer_type = "Activation"
        self.trainable = False
        self.logging = logging

    def forward(self, X):
        if self.activation == "relu":
            self.A = gates.MatReLUGate()
        elif self.activation == "sigmoid":
            self.A = gates.MatSigmoidGate()
        elif self.activation == "softmax":
            self.A = gates.SoftmaxGate()
            
        A_out = self.A.forward(X)

        if self.logging:
            print("[+] ACTIVATION_OUT:", A_out.shape)

        return A_out

    def backward(self, dZ):
        dA = self.A.backward(dZ)
        return dA


class Flatten():
    def __init__(self, new_shape, logging=False):
        self.new_shape = new_shape
        self.layer_type = "Flatten"
        self.trainable = False
        self.logging = logging

    def forward(self, X):
        self.old_shape = X.shape
        F = X.reshape(self.new_shape)

        if self.logging:
            print("[+] FLATTEN_OUT:", F.shape)

        return F

    def backward(self, dZ):
        return dZ.reshape(self.old_shape)



class CrossEntropy():    
    def forward(self, Y, T):
        self.L = gates.CrossEntropyLossGate()
        L_out = self.L.forward(Y, T)
        return L_out

    def backward(self):
        return self.L.backward()


class MeanSquared():    
    def forward(self, Y, T):
        self.L = gates.MSEGate()
        L_out = self.L.forward(Y, T)
        return L_out

    def backward(self):
        return self.L.backward()