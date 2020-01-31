import numpy as np

def im2col(X, X_n, X_c, W_n, W_c, stride):
    im2col_cols = W_c * W_n * W_n
    im2col_rows = X_n * X_n
    window_col = np.zeros(shape=(im2col_rows, im2col_cols), dtype=np.int)

    i = 0

    for row in range(0, X_n, stride):
        for col in range(0, X_n, stride):
            window_col[i] = get_window(X, W_n, W_c, row, col).flatten()
            i += 1

    return window_col.T


def get_window(X, W_n, W_c, row, col):
    window = np.zeros(shape=(W_c, W_n, W_n), dtype=np.int)

    for c in range(W_c):
        window[c] = X[c][row:row + W_n, col:col + W_n]

    return window


def get_window2d(X, W_n, row, col):
    return X[row:row + W_n, col:col + W_n]


def get_shape(X):
    X_c = 1
    N = 1

    try:
        N, X_c, X_h, X_w = X.shape
    except ValueError:
        X_c, X_h, X_w = X.shape

    assert X_h == X_w, "[!] ERROR: X width and height are not equal"

    return N, X_h, X_c


def add_padding(X, padding):
    X_n, X_c = get_shape(X)
    new_X = np.zeros(shape=(X_c, X_n + 2 * padding, X_n + 2 * padding), dtype=np.int)

    for c in range(X_c):
        new_X[c] = np.pad(X[c], padding, mode="constant")

    return new_X


def add_same_padding(X, W_n, W_c, stride):
    X_n, X_c = get_shape(X)
    padding = get_same_padding(X_n, W_n, stride)

    new_X_n, new_X_c = get_new_shape(X, W_n, W_c, padding, stride)

    assert X_n == new_X_n, "[!] ERROR: Same pad is impossible"

    return add_padding(X, padding)


def get_same_padding(X_n, W_n, stride):
    return int(np.floor((X_n * (stride - 1) - stride + W_n)/2))


def get_new_shape(X, W_n, W_c, padding, stride):
    X_n, X_c = get_shape(X)

    new_X_n = int(np.floor((X_n + 2 * padding - W_n)/stride) + 1)

    return new_X_n, X_c