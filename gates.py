import numpy as np
import utils
import standfordutils

class MatMultGate():
    def forward(self, A, B):
        self.A = A
        self.B = B
        return np.dot(self.A, self.B)

    def backward(self, dZ):
        return np.dot(dZ, self.B.T), np.dot(self.A.T, dZ)


class MatAddGate():
    def forward(self, A, B):
        self.A = A
        self.B = B
        return self.A + self.B

    def backward(self, dZ):
        return dZ, np.sum(dZ, axis=1, keepdims=True)


class ConvMultGate():
    def forward(self, A, B):
        self.A = A
        self.B = B
        return np.dot(self.A, self.B)

    def backward(self, dZ):
        dA = np.sum(np.dot(dZ, self.B.transpose(0,2,1)), axis=(0,2))
        dB = np.dot(self.A.transpose(1,0), dZ).transpose(1,0,2)
        return dA, dB 


class ConvAddGate():
    def forward(self, A, B):
        self.A = A
        self.B = B
        return (self.A + self.B).transpose(1,0,2)

    def backward(self, dZ):
        return dZ, np.sum(dZ, axis=(0,2), keepdims=True).transpose(1,0,2)


class MatSigmoidGate():
    def sigmoid(self, A):
        return 1.0/(1.0 + np.exp(-A))

    def forward(self, A):
        self.A = A
        F = self.sigmoid(self.A)
        self.dA = F*(1-F)
        return F

    def backward(self, dZ):
        return dZ*self.dA


class MatLeakyReLUGate():
    def relu(self, A, alpha=0.1):
        A = np.where(A > 0, A, A * alpha)
        return A

    def relu_deriv(self, A, alpha=0.1):
        dA = np.ones_like(A)
        dA[A<0] = alpha
        return dA

    def forward(self, A):
        self.A = A
        F = self.relu(self.A)
        self.dA = self.relu_deriv(self.A)
        return F

    def backward(self, dZ):
        return dZ*self.dA


class MatReLUGate():
    def relu(self, A):
        A[A<0] = 0
        return A

    def forward(self, A):
        self.A = A
        F = self.relu(self.A)
        self.dA = (F > 0) * 1
        return F

    def backward(self, dZ):
        return dZ*self.dA


class MSEGate():
    def forward(self, Y, T):
        self.Y = Y
        self.T = T
        self.n_labels, self.n_examples = self.Y.shape
        loss = (self.T-self.Y)**2
        loss = np.sum(loss, axis=0, keepdims=True)/self.n_labels
        loss = np.sum(loss)/self.n_examples
        self.dY = -(self.T-self.Y)/self.n_examples
        return loss

    def backward(self):
        return self.dY  


class SoftmaxGate():
    def softmax(self, A):
        A -= np.max(A)
        A_exp = np.exp(A) + 1e-12
        return  A_exp/np.sum(A_exp, axis=0, keepdims=True) 

    def forward(self, A):
        self.A = A
        _, self.n_examples = self.A.shape
        F = self.softmax(self.A)
        return F

    def backward(self, dZ):
        return dZ


class CrossEntropyLossGate():
    def cross_entropy(self, Y, T):
        log_likelihood = -np.log(Y[T.argmax(axis=0), range(self.n_examples)])
        #print("LOG_LIKELIHOOD:", log_likelihood)
        loss = np.average(log_likelihood)
        return loss

    def forward(self, Y, T):
        _, self.n_examples = T.shape
    
        f = self.cross_entropy(Y, T)

        self.dF = np.copy(Y)
        self.dF[T.argmax(axis=0), range(self.n_examples)] -= 1
        self.dF /= self.n_examples

        return f

    def backward(self):
        return self.dF


class MaxPoolingGate():
    def max_pool(self, X, W_n, stride):
        N, C, H, W = X.shape
        pool_height, pool_width = W_n, W_n

        self.W_n = W_n
        self.stride = stride

        assert (H - pool_height) % stride == 0, 'Invalid height'
        assert (W - pool_width) % stride == 0, 'Invalid width'

        out_height = int((H - pool_height) / stride + 1)
        out_width = int((W - pool_width) / stride + 1)

        x_split = X.reshape(N * C, 1, H, W)
        _, _, self.x_cols = standfordutils.im2col_indices(x_split, pool_height, pool_width, padding=0, stride=stride)
        self.x_cols_argmax = np.argmax(self.x_cols, axis=0)
        x_cols_max = self.x_cols[self.x_cols_argmax, np.arange(self.x_cols.shape[1])]
        out = x_cols_max.reshape(out_height, out_width, N, C).transpose(2, 3, 0, 1)

        return out

    def forward(self, X, window_size, stride):
        self.X = X
        self.F = self.max_pool(X, window_size, stride)
        return self.F

    def backward(self, dout):
        N, C, H, W = self.X.shape
        pool_height, pool_width = self.W_n, self.W_n
        stride = self.stride

        dout_reshaped = dout.transpose(2, 3, 0, 1).flatten()
        dx_cols = np.zeros_like(self.x_cols)
        dx_cols[self.x_cols_argmax, np.arange(dx_cols.shape[1])] = dout_reshaped
        dx = standfordutils.col2im_indices(dx_cols, (N * C, 1, H, W), pool_height, pool_width, padding=0, stride=stride)
        dx = dx.reshape(self.X.shape)

        return dx

    