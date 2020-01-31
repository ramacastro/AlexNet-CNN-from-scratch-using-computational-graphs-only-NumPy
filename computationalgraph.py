import os
import shelve
import layers
import numpy as np

class ComputationalGraph():
    def __init__(self, name, loss_function="cross-entropy", lr=1e-3, reg=1e-4):
        self.name = name
        self.loss_function = loss_function
        self.lr = lr
        self.reg = reg
        self.graph = []
        self.regularize = self.reg>0
        if self.regularize:
            self.regularization_losses = []
        

    def add_layer(self, layer):
        if layer.layer_type == "FullyConnected" and self.regularize:
            layer.add_regularization(self.reg)
        self.graph.append(layer)

    def predict(self, X):
        output = X 
        if self.regularize:
            self.regularization_losses = []

        for layer in self.graph:
            output = layer.forward(output)
            if layer.layer_type == "FullyConnected" and self.regularize:
                self.regularization_losses.append(layer.regularization_loss())
            
        return output

    def compute_loss(self, Y, T):
        if self.loss_function == "cross-entropy":
            self.loss_layer = layers.CrossEntropy()
        elif self.loss_function == "mean-squared":
            self.loss_layer = layers.MeanSquared()
        data_loss = self.loss_layer.forward(Y, T)
        loss = data_loss
        _, self.n_examples = T.shape

        if self.regularize:
            regularization_loss = np.sum(self.regularization_losses)*(self.reg/self.n_examples)
            loss += regularization_loss
            
        return loss

    def train(self, X, T):
        Y = self.predict(X)
        loss = self.compute_loss(Y, T)
        dZ = self.loss_layer.backward()

        for layer in reversed(self.graph):
            dZ = layer.backward(dZ)
            if layer.trainable:
                layer.update_parameters(self.lr)

        predicted_ok = np.sum(np.argmax(Y, axis=0) == np.argmax(T, axis=0))

        return Y, loss, predicted_ok

    def save_parameters(self):
        shelve_filename = self.name + "_trained_parameters.db"

        parameters_shelve = shelve.open(shelve_filename, writeback=True)

        parameters_shelve["parameters"] = []
        
        for layer in self.graph:
            if layer.trainable:
                layer_parameters = {"W":layer.W, "B":layer.B}
                parameters_shelve["parameters"].append({"layer_name":layer.name, "layer_parameters":layer_parameters})

        parameters_shelve.sync()
        parameters_shelve.close()
        print("\n[+] " + self.name + " parameters saved successfully")
    
    def load_parameters(self):
        parameters_shelve = shelve.open(self.name + "_trained_parameters.db", writeback=True)

        trainable_layer = 0

        for i in range(len(self.graph)):
            layer = self.graph[i]
            if layer.trainable:
                layer_info = parameters_shelve["parameters"][trainable_layer]
                layer_name = layer_info["layer_name"]
                layer_parameters = layer_info["layer_parameters"]
                print("[+] loading layer", layer_name, end=": ")
                W = layer_parameters["W"]
                B = layer_parameters["B"]
                print("W:", W.shape, "| B:", B.shape)
                layer.set_parameters(W, B)
                trainable_layer += 1

        parameters_shelve.close()
        print("\n[+] " + self.name + " parameters loaded successfully")