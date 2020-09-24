import json
import sys
import json
import random
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#定义激活函数类
class Activation_Function(object):
    class sigmoid(object):
        @staticmethod
        def fn(z):
            return 1.0 / (1.0 + np.exp(-z))

        @staticmethod
        def prime(z):
            return Activation_Function.sigmoid.fn(z) * (1 - Activation_Function.sigmoid.fn(z))

    class softmax(object):
        @staticmethod
        def fn(z):
            z -= np.max(z)
            sm = (np.exp(z).T / np.sum(np.exp(z), axis=1))
            return sm

    class relu(object):
        @staticmethod
        def fn(z):
            z = (z + np.abs(z)) / 2.0
            return z

        @staticmethod
        def prime(z):
            z[z <= 0] = 0
            z[z > 0] = 1
            return z

#定义损失函数类
class cost_Function(object):
    class QuadraticCost(object):
        @staticmethod
        def fn(a, y):
            return 0.5 * np.linalg.norm(a - y) ** 2

    class CrossEntropyCost(object):
        @staticmethod
        def delta(z, a, y):
            return (a - y)

        @staticmethod
        def fn(a, y):
            return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

#定义全连接层,需要切换激活函数，请修改activate为激活函数类中的任一一个，也可自行添加激活函数
class FullyConnectedLayer(object):
    @staticmethod
    def feedforward(a, w, b, activate=Activation_Function.relu):
        z = np.dot(w, a) + b
        a = activate.fn(z)
        return z, a

    @staticmethod
    def backprop(z, a, w, delta, activate=Activation_Function.relu):
        sp = activate.prime(z)
        delta = np.dot(w.transpose(), delta) * sp
        nabla_b = delta
        nabla_w = np.dot(delta, a.transpose())
        return sp,nabla_b, nabla_w, delta


class Network(object):

    def __init__(self, sizes, cost=cost_Function.CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost = cost
        self.layers_bincode = []
        self.training_loss=[]
        self.Pl_z=[]
        self.Pl_w=[]
        self.Pl_b=[]
        self.Pa_z=[]
        self.Pz_w=[]
        self.Pl_a=[]
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]


    def SGD(self,p, training_data, epochs, mini_batch_size, eta,
            evaluation_data=None,
            monitor_evaluation_cost=True,
            monitor_evaluation_accuracy=True,
            monitor_training_cost=True,
            monitor_training_accuracy=True,
            early_stopping_n=0):

        # early stopping functionality:
        best_accuracy = 1   
        training_data = list(training_data)
        n = len(training_data)

        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)

        # early stopping functionality:
        best_accuracy = 0
        no_accuracy_change = 0

        evaluation_cost, evaluation_accuracy,training_cost, training_accuracy ,training_loss,layers_bincode= [], [],[], [],[],[]
        Pl_z,Pl_w,Pl_b,Pz_w,Pa_z,Pl_a=[],[],[],[],[],[]
        for j in range(epochs):
            # random.shuffle(training_data)
            mini_batches =  [training_data[k:k + mini_batch_size]  for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                    self.update_mini_batch( p,mini_batch, eta,  len(training_data))
            Pl_z.append(self.Pl_z)
            Pl_w.append(self.Pl_w)
            Pl_b.append(self.Pl_b)
            Pz_w.append(self.Pz_w)
            Pa_z.append(self.Pa_z)
            layers_bincode.append(self.layers_bincode)
            self.Pl_z,self.Pl_w,self.Pl_b,self.Pa_z,self.Pz_w,self.Pl_a=[],[],[],[],[],[]
            self.layers_bincode=[]


            print("Epoch %s training complete" % j)

            if monitor_training_cost:
                cost_list,cost = self.total_cost(training_data,)
                training_loss.append(cost_list)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data)[-1]
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data), n_data))
            # plt.show()
            # Early stopping:
            if early_stopping_n > 0:
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    no_accuracy_change = 0
                    # print("Early-stopping: Best so far {}".format(best_accuracy))
                else:
                    no_accuracy_change += 1

                if (no_accuracy_change == early_stopping_n):
                    # print("Early-stopping: No accuracy change in last epochs: {}".format(early_stopping_n))
                    return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy
        return evaluation_cost, evaluation_accuracy, \
               training_cost, training_accuracy,\
               training_loss, layers_bincode, Pl_z ,\
               Pl_w ,  Pl_b  ,Pa_z  ,Pz_w\
 
    def update_mini_batch(self,p, mini_batch, eta, n):   
        self.nabla_b = [np.zeros(b.shape) for b in self.biases]
        self.nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            self.nabla_b = [nb + dnb for nb, dnb in zip(self.nabla_b, delta_nabla_b)]
            self.nabla_w = [nw + dnw for nw, dnw in zip(self.nabla_w, delta_nabla_w)]
        for x in p:
            layers_bincode=self.feedforward(np.expand_dims(x,axis=-1))[0]
            self.layers_bincode.append(layers_bincode)



        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, self.nabla_w)]

        self.biases = [b - (eta / len(mini_batch)) * nb
                        for b, nb in zip(self.biases, self.nabla_b)]

    def feedforward(self, activation):
        activations = [activation]  
        zs = []  
        layers_bincode = []
        for i in range(self.num_layers - 2):
            z, activation = FullyConnectedLayer.feedforward(activation, self.weights[i], self.biases[i],
                                                            activate=Activation_Function.relu)
            layer_bincode = np.where(activation > 0, 1, 0)
            zs.append(z)
            activations.append(activation)
            layers_bincode.append(layer_bincode)
        z, activation = FullyConnectedLayer.feedforward(activation, self.weights[-1], self.biases[-1],
                                                        activate=Activation_Function.sigmoid)
        layer_bincode = np.where(activation > 0.5, 1, 0)
        zs.append(z)
        activations.append(activation)
        layers_bincode.append(layer_bincode)
        return layers_bincode, zs, activations

    def backprop(self, x, y):
        # feedforward
        Pl_z=[]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # network
        layers_bincode, zs, activations = self.feedforward(x)
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        # 保存所有delta
        # 保存所有节点的激活情况和激活值
        Pl_z.append(delta)
        self.Pz_w.append(activations)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            sp,nabla_b[-l], nabla_w[-l], delta = FullyConnectedLayer.backprop(zs[-l], activations[-l - 1],
                                                                           self.weights[-l + 1], delta,
                                                                activate=Activation_Function.relu)
            Pl_z.append(delta)
        self.Pl_z.append(Pl_z[::-1])
        self.Pa_z.append(sp)    
        self.Pl_w.append(nabla_w)
        self.Pl_b.append(nabla_b)
        return  nabla_b, nabla_w

    def accuracy(self, data):
        results = [(self.feedforward(x)[-1][-1], y)
                    for (x, y) in data]
        result_accuracy = sum(int(round(x[0][0]) == y[0]) for (x, y) in results)
        return result_accuracy

    def total_cost(self, data):
        cost = 0.0
        cost_list=[]
        for x, y in data:
            a = self.feedforward(x)[-1][-1]
            cost_list.append(self.cost.fn(a, y))
            cost += self.cost.fn(a, y) / len(data)
        return cost_list,cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "nabla_w": [w.tolist() for w in self.nabla_w],
                "nabla_b": [b.tolist() for b in self.nabla_b],
                "cost": str(self.cost)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

    def plot_training_cost(self, training_cost, num_epochs, training_cost_xmin):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.arange(training_cost_xmin, num_epochs),
                training_cost[training_cost_xmin:num_epochs],
                color='#2A6EA6')
        ax.set_xlim([training_cost_xmin, num_epochs])
        ax.grid(True)
        ax.set_xlabel('Epoch')
        ax.set_title('Cost on the training data')
        return fig

    def plot_test_accuracy(self, test_accuracy, num_epochs, test_accuracy_xmin):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.arange(test_accuracy_xmin, num_epochs),
                [accuracy / 100.0
                 for accuracy in test_accuracy[test_accuracy_xmin:num_epochs]],
                color='#2A6EA6')
        ax.set_xlim([test_accuracy_xmin, num_epochs])
        ax.grid(True)
        ax.set_xlabel('Epoch')
        ax.set_title('Accuracy (%) on the test data')
        return fig

    def plot_test_cost(self, test_cost, num_epochs, test_cost_xmin):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.arange(test_cost_xmin, num_epochs),
                test_cost[test_cost_xmin:num_epochs],
                color='#2A6EA6')
        ax.set_xlim([test_cost_xmin, num_epochs])
        ax.grid(True)
        ax.set_xlabel('Epoch')
        ax.set_title('Cost on the test data')
        return fig

    def plot_training_accuracy(self, training_accuracy, num_epochs,
                               training_accuracy_xmin, training_set_size):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.arange(training_accuracy_xmin, num_epochs),
                [accuracy * 100.0 / training_set_size
                 for accuracy in training_accuracy[training_accuracy_xmin:num_epochs]],
                color='#2A6EA6')
        ax.set_xlim([training_accuracy_xmin, num_epochs])
        ax.grid(True)
        ax.set_xlabel('Epoch')
        ax.set_title('Accuracy (%) on the training data')
        return fig

    def accuracy_plot_overlay(self, test_accuracy, training_accuracy, num_epochs, 
                     training_set_size,test_set_size):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.arange(0, num_epochs),
                [accuracy * 100.0  /test_set_size
                 for accuracy in test_accuracy],
                color='#2A6EA6',
                label="Accuracy on the test data")
        ax.plot(np.arange(0, num_epochs),
                [accuracy * 100.0 / training_set_size
                 for accuracy in training_accuracy],
                color='#FFA933',
                label="Accuracy on the training data")
        ax.grid(True)
        ax.set_xlim([0, num_epochs])
        ax.set_xlabel('Epoch')
        #     ax.set_ylim([90, 100])
        plt.legend(loc="lower right")
        return fig


    def cost_plot_overlay(self, test_cost, train_cost, num_epochs):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.arange(0, num_epochs),
                train_cost[0:num_epochs],
                color='#FFA933',
                label="cost on the training data")
        ax.plot(np.arange(0, num_epochs),
                test_cost[0:num_epochs],
                color='#2A6EA6',
                label="cost on the test data")
        ax.grid(True)
        ax.set_xlim([0, num_epochs])
        ax.set_xlabel('Epoch')
        plt.legend(loc="lower right")
        return fig

def scatter(p, c, X, y, cm_bright, wb=None, wb_grand=None, cmap='coolwarm'):
    cols = p.shape[-1]
    assert cols in (1, 2, 3)
    fig = plt.figure(figsize=(6, 4))
    if cols == 3:
        ax3d = Axes3D(fig)
        if wb is not None:
            a1, a2 = p.min(0)[:2]
            b1, b2 = p.max(0)[:2]
            a, b = np.mgrid[a1 - 1:b1:10j, a2 - 1:b2:10j]
            (u1, u2, u3), b_ = wb
            # ax3d.plot([0, u1], [0, u2], [0, u3], 'r--')
            z_ = (a * u1 + b * u2 + b_) / (-u3)
            ax3d.plot_wireframe(a, b, z_)
        if wb_grand is not None:
            (w1, w2, w3), b_1 = wb_grand
            # ax3d.plot([u1, (u1 + w1)], [u2, (u2 + w2)], [u3, (u3 + w3)], 'g--')
            # ax3d.plot([0, (u1 + w1)], [0, (u2 + w2)], [0, (u3 + w3)], 'b--')
            z_ = (a * u1  + b * u2  + +b_+b_1) / (-u3 )
            ax3d.plot_wireframe(a, b, z_)
            z_ = (a * (u1 + w1) + b * (u2 + w2) + b_1) / (-(u3 + w3))
            ax3d.plot_wireframe(a, b, z_)

        point = ax3d.scatter(*X.T, c=y, cmap=cm_bright, edgecolors='white', s=40, linewidths=0.5)
        mp = ax3d.scatter(*p.T, c=c, cmap=cmap)
        fig.colorbar(mp, shrink=0.8)
        fig.colorbar(point, shrink=0.8)

        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('Z')
        return ax3d

    elif cols == 2:
        ax = plt.gca()
        ax.axis('equal')
        if wb is not None:
            a1, a2 = p.min(0) - 0.2
            b1, b2 = p.max(0) + 0.2
            (w1, w2), b_ = wb
            # ax.plot([0, w1 / 2], [0, w2 / 2], 'r--')
            y1, y2 = (a1 * w1 + b_) / (-w2), (b1 * w1 + b_) / (-w2)
            ax.plot([a1, b1], [y1, y2], 'r--')
            ax.set_ylim(a2, b2)
        if wb_grand is not None:
            (u1, u2), b_ = wb_grand
            ax.plot([a1, b1], [y1+b_, y2+b_], 'b--')
            # ax.plot([w1 / 2, (u1 + wb[0][0]) / 2], [w2 / 2, u2 / u1 * (u1 + wb[0][0]) / 2 + w2 / 2 - u2 * w1 / u1 / 2],
            #         'b--')
            (u1, u2) = (u1, u2) + wb[0]
            b_ = b_ + wb[1]
            # ax.plot([0, u1 / 2], [0, u2 / 2], 'g--')
            y1, y2 = (a1 * u1 + b_) / (-u2), (b1 * u1 + b_) / (-u2)
            ax.plot([a1, b1], [y1, y2], 'g--')
            ax.set_ylim(a2, b2)

        st = ax.scatter(*p.T, c=c, cmap=cmap)
        point = ax.scatter(*X.T, c=y, alpha=1, cmap=cm_bright, edgecolors='white', s=20, linewidths=0.5)
        fig.colorbar(point)
        fig.colorbar(st)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')


    else:
        ax = plt.gca()
        t, tt = np.zeros_like(p.flat), np.zeros_like(X.flat)
        st = plt.scatter(p.flat, t, c=c, cmap=cmap)
        point = ax.scatter(X.flat, tt, c=y, alpha=1, cmap=cm_bright, edgecolors='white', s=20, linewidths=0.5)
        fig.colorbar(st)
        fig.colorbar(point)
    # 3D绘图加equal会出现问题
    # plt.axis('equal')
    # plt.tight_layout()
    return ax


def mapping(code):
    numMap = np.zeros(code.shape[0])
    uniq = np.unique(code, axis=0)
    for i, arr in enumerate(uniq):
        m = (np.sum(code == arr, axis=1) == code.shape[-1])
        numMap[m] = i
    return numMap


#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


#### Miscellaneous functions
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
