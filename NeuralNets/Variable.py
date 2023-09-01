import math
import random


class Variable:
    def __init__(self, data, _children=()):
        self.data = data
        self.children = set(_children)
        self.lable = ""
        self._backward = lambda: None
        self.grade = 0.0

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Variable(other)
        out = Variable(self.data + other.data, (self, other))

        def backward():
            self.grade += out.grade
            other.grade += out.grade
        out._backward = backward
        return out

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Variable(other)
        out = Variable(self.data * other.data, (self, other))

        def backward():
            self.grade += other.data*out.grade
            other.grade += self.data*out.grade
        out._backward = backward
        return out

    def __truediv__(self, other):
        return self * other**-1

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __radd__(self, other):  # other + self
        return self + other

    def __rtruediv__(self, other):  # other / self
        return other * self**-1

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            out = Variable(self.data**other, (self,))

            def backward():
                self.grade += (other*self.data**(other-1))*out.grade

            out._backward = backward
            return out
        else:
            raise ValueError("Invalid value in pow opration")
            return null

    def tanh(self):
        out = Variable((math.exp(2*self.data) - 1) /
                       (math.exp(2*self.data) + 1), (self,))

        def backward():
            self.grade = (1-out.data*out.data)*out.grade

        out._backward = backward
        return out

    def exp(self):
        out = Variable(math.exp(self.data), (self,))

        def backward():
            self.grade += out.data*out.grade
        out._backward = backward
        return out

    def relu(self):
        out = Variable(self.data, (self,))
        if out.data < 0:
            out.data = 0

        def backward():
            if out.data > 0:
                self.grade = out.grade
            else:
                self.grade = 0
        out._backward = backward
        return out

    def __repr__(self):
        return f'Variable({self.data} and grade:{self.grade})'

    def backpropogation(self):
        self.grade = 1.0
        visited = []
        topological = []

        def dfs(node):
            visited.append(node)
            for child in node.children:
                if child not in visited:
                    dfs(child)
            topological.append(node)

        dfs(self)
        for node in reversed(topological):
            node._backward()


class Neurons:
    def __init__(self, num_weights):
        self.weigths = [Variable(random.uniform(1, -1))
                        for _ in range(num_weights)]
        self.b = Variable(0)

    def parameters(self):
        return self.weigths + [self.b]

    def __call__(self, values):
        var = sum((wi*xi for wi, xi in zip(self.weigths, values)), self.b)
        return var


class Layer:
    def __init__(self, num_neurons, num_weights, activation='relu'):
        self.neurons = [Neurons(num_weights) for _ in range(num_neurons)]
        self.activation = activation

    def __call__(self, values):
        output = [ni(values) for ni in self.neurons]
        if self.activation == 'relu':
            output = [out.relu() for out in output]
        elif self.activation == 'tanh':
            output = [out.tanh() for out in output]
        return output

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class Perceptron:
    def __init__(self, num_layer, tup):
        self.layers = [Layer(tup[i][0], tup[i][1]) for i in range(num_layer)]

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __call__(self, values):
        inputs = values
        for L in self.layers:
            outputs = L(inputs)
            inputs = outputs

        return inputs[0]
