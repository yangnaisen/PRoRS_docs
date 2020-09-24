
import json
import random
import sys

# Third-party libraries
import numpy as np

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.
        MSE
        """
        return 0.5*np.linalg.norm(a-y,ord=2)

    @staticmethod
    def delta(z, a, y, activation_prime):
        """Return the error delta from the output layer."""
        return (a-y) * activation_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))


    @staticmethod
    def delta(z, a, y, activation_prime):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return (a-y)*activation_prime(z)/sigmoid_prime(z)

class BinaryLogCost(object):
    @staticmethod
    def fn(a, y):
        a = np.clip(a,1e-10, 1-1e-10)
        return -y*np.log(a) - (1-y)*np.log(1-a)

    @staticmethod
    def delta(z, a, y, activation_prime):

        return (a-y)*activation_prime(z)/sigmoid_prime(z)
        # return a - y

class sigmoid_activation(object):
    @staticmethod
    def fn(z):
        return sigmoid(z)

    @staticmethod
    def prime(z):
        return sigmoid_prime(z)

class relu_activation(object):
    @staticmethod
    def fn(z):
        return relu(z)

    @staticmethod
    def prime(z):
        return relu_prime(z)

class tanh_activation(object):
    @staticmethod
    def fn(z):
        return np.tanh(z)
    @staticmethod
    def prime(z):
        return 1-np.tanh(z)**2



#### Main Network class
class Network(object):

    def __init__(self, sizes,layers_type, activation_fn =sigmoid_activation, cost=CrossEntropyCost, regularizer='L2'):
        """全连接与卷积输入如[2,3,[3,2],1],
        其中的[3,2]代表卷积核为1*2,个数为卷积核数量为3"""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost=cost
        self.regularizer = regularizer
        self.activation_fn = activation_fn
        self.class_activation = sigmoid_activation
        self.weights_log = []
        self.biases_log = []
        self.w_grad_log = []
        self.b_grad_log = []
        self.train_loss_log=[]
        self.test_loss_log=[]
        self.delta_log = []
        self.Pl_Pa_log = []
        self.layers_type = layers_type
        self.pooling_ind = []
        self.a_pool_before = []

    def set_weights(self,weights,bias):
        self.weights = weights
        self.biases = bias

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.
        """
        self.weights = []
        self.biases = []
        for x,y in zip(self.sizes[:-1],self.sizes[1:]):
            if isinstance(y,int) and isinstance(x,int):
                self.weights.append(np.random.randn(y,x)/np.sqrt(x))
                self.biases.append(np.random.randn(y,1))
            elif isinstance(y,list) and isinstance(x,int):
                assert len(y)==2
                self.weights.append(np.random.randn(*y)/np.sqrt(x))
                self.biases.append(np.random.randn(y[0],1))
            elif isinstance(x,list) and isinstance(y,int):
                assert  len(x)==2
                self.weights.append(np.random.randn(y,x[0])/np.sqrt(x[0]))
                self.biases.append(np.random.randn(y,1))
            elif isinstance(x,list) and isinstance(y,list):
                assert  len(x)==len(y)==2
                self.weights.append(np.random.randn(*y)/np.sqrt(x[0]))
                self.biases.append(np.random.randn(y[0], 1))
            else:
                raise TypeError

    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.
        """

        self.weights = []
        self.biases = []
        for x, y in zip(self.sizes[:-1], self.sizes[1:]):
            if isinstance(y, int) and isinstance(x, int):
                self.weights.append(np.random.randn(y, x))
                self.biases.append(np.random.randn(y, 1))
            elif isinstance(y, list) and isinstance(x, int):
                assert len(y) == 2
                self.weights.append(np.random.randn(*y))
                self.biases.append(np.random.randn(y[0], 1))
            elif isinstance(x, list) and isinstance(y, int):
                assert len(x) == 2
                self.weights.append(np.random.randn(y, x[0]))
                self.biases.append(np.random.randn(y, 1))
            elif isinstance(x, list) and isinstance(y, list):
                assert len(x) == len(y) == 2
                self.weights.append(np.random.randn(*y))
                self.biases.append(np.random.randn(y[0], 1))
            else:
                raise TypeError

    def large_weight_ones(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.
        """

        self.weights = []
        self.biases = []
        for x, y in zip(self.sizes[:-1], self.sizes[1:]):
            if isinstance(y, int) and isinstance(x, int):
                self.weights.append(np.ones((y, x)))
                self.biases.append(np.ones((y, 1)))
            elif isinstance(y, list) and isinstance(x, int):
                assert len(y) == 2
                self.weights.append(np.ones((y[0],y[1])))
                self.biases.append(np.ones((y[0], 1)))
            elif isinstance(x, list) and isinstance(y, int):
                assert len(x) == 2
                self.weights.append(np.ones((y, x[0])))
                self.biases.append(np.ones((y, 1)))
            elif isinstance(x, list) and isinstance(y, list):
                assert len(x) == len(y) == 2
                self.weights.append(np.ones((y[0],y[1])))
                self.biases.append(np.ones((y[0], 1)))
            else:
                raise TypeError

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for l,(b,w) in enumerate(zip(self.biases[:-1],self.weights[:-1])):
            if isinstance(self.sizes[l+1],int):
                a=self.activation_fn.fn(np.dot(w,a)+b)
            else:
                if isinstance(self.sizes[l],list):
                    in_n = self.sizes[l][0]
                else:
                    in_n = self.sizes[l]
                a_temp=np.zeros(shape=(self.sizes[l+1][0],in_n-self.sizes[l+1][1]+1))
                for c_num in range(in_n-self.sizes[l+1][1]+1):
                    a_temp[:,c_num] = np.squeeze(self.activation_fn.fn(np.dot(w,a[c_num:c_num+self.sizes[l+1][1]])+b))
                a=np.expand_dims(np.max(a_temp,axis=1),1)
        a = self.class_activation.fn(np.dot(self.weights[-1], a) + self.biases[-1])
        return a



    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None, #评测数据
            monitor_evaluation_cost=False, #监测验证损失
            monitor_evaluation_accuracy=True, #监测验证精度
            monitor_training_cost=True, #监测训练损失
            monitor_training_accuracy=True, #监测训练精度
            early_stopping_n = 0, #早停阈值
            verbose = 0, # 是否开启冗余输出，开启之后才能显示上述监测内容
            save_loss=False, # 保存损失
            save_delta = False, # 保存误差
            save_grad = False,  # 保存梯度
            save_weights = False, # 保存训练过程中的权重
            save_Pl_Pa = False): #保存各层偏导
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.

        """
        self.weights_log.append(self.weights)
        self.biases_log.append(self.biases)


        # early stopping functionality:
        best_accuracy=1

        training_data = list(training_data)
        training_data_raw = training_data.copy()
        n = len(training_data)

        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)

        # early stopping functionality:
        best_accuracy=0
        no_accuracy_change=0

        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            # random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))

            self.train_loss_log.append(self.get_loss(training_data_raw,lmbda))
            self.test_loss_log.append(self.get_loss(evaluation_data,lmbda))
            if verbose:
                print("Epoch %s training complete" % j)

                if monitor_training_cost:
                    cost = self.total_cost(training_data, lmbda)
                    training_cost.append(cost)
                    print("Cost on training data: {}".format(np.squeeze(cost)))
                if monitor_training_accuracy:
                    accuracy = self.accuracy(training_data)
                    training_accuracy.append(accuracy)
                    print("Accuracy on training data: {} / {}".format(accuracy, n))
                if monitor_evaluation_cost:
                    cost = self.total_cost(evaluation_data, lmbda, convert=True)
                    evaluation_cost.append(cost)
                    print("Cost on evaluation data: {}".format(cost))
                if monitor_evaluation_accuracy:
                    accuracy = self.accuracy(evaluation_data)
                    evaluation_accuracy.append(accuracy)
                    print("Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data), n_data))

            # Early stopping:
            if early_stopping_n > 0:
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    no_accuracy_change = 0
                    #print("Early-stopping: Best so far {}".format(best_accuracy))
                else:
                    no_accuracy_change += 1

                if (no_accuracy_change == early_stopping_n):
                    #print("Early-stopping: No accuracy change in last epochs: {}".format(early_stopping_n))
                    return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy
        if save_loss:
            np.save('log/train_loss.npy',np.squeeze(self.train_loss_log))
            np.save('log/test_loss.npy',np.squeeze(self.test_loss_log))
        if save_weights:
            np.save('log/weights_log.npy',self.weights_log)
            np.save('log/bias_log.npy',self.biases_log)
        if save_delta:
            np.save('log/delta_log.npy',self.delta_log)
        if save_grad:
            np.save('log/w_grad_log.npy',self.w_grad_log)
            np.save('log/b_grad_log.npy',self.b_grad_log)
        if save_Pl_Pa:
            np.save('log/Pl_Pa.npy',self.Pl_Pa_log)

        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.
        最小批反向传播过程
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        delta = []
        pl_pa_log = []
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            delta.append(delta_nabla_b)
            pl_pa_log.append(self.Pl_Pa.copy())
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        #记录偏差值，结构为batch-sample-delta
        self.delta_log.append(delta)
        self.Pl_Pa_log.append(pl_pa_log)
        self.w_grad_log.append([nw/len(mini_batch) for nw in nabla_w])
        self.b_grad_log.append([nb/len(mini_batch) for nb in nabla_b])
        if self.regularizer=='L2':
            self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                            for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b-(eta/len(mini_batch))*nb
                           for b, nb in zip(self.biases, nabla_b)]
        elif self.regularizer=='L1':
            self.weights = [(1 - eta * (lmbda / n)) * np.sign(w) - (eta / len(mini_batch)) * nw
                            for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - (eta / len(mini_batch)) * nb
                           for b, nb in zip(self.biases, nabla_b)]
        else:
            print('Invalid regularizer!')
        self.weights_log.append(self.weights)
        self.biases_log.append(self.biases)


    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        pooling_ind = []
        self.Pl_Pa=[]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for l,(b,w) in enumerate(zip(self.biases[:-1],self.weights[:-1])):
            if isinstance(self.sizes[l+1],int):
                z = np.dot(w, activation) + b
                zs.append(z)
                activation=self.activation_fn.fn(z)
                activations.append(activation)
                pooling_ind.append(0)
            else:
                if isinstance(self.sizes[l],list):
                    in_n = self.sizes[l][0]
                else:
                    in_n = self.sizes[l]
                a_temp=np.zeros(shape=(self.sizes[l+1][0],in_n-self.sizes[l+1][1]+1))
                zz = np.zeros_like(a_temp)
                a_2_cnn = activation
                for c_num in range(in_n-self.sizes[l+1][1]+1):
                    #c_num为卷积出的特征维度
                    z = np.dot(w,activation[c_num:c_num+self.sizes[l+1][1]])+b
                    zz[:,c_num] = np.squeeze(z)
                    a_temp[:,c_num] = np.squeeze(self.activation_fn.fn(z))
                # self.a_pool_before.append(a_temp)
                pooling_ind.append(np.argmax(a_temp,axis=1))
                z = np.expand_dims(zz[list(range(len(zz))),np.argmax(zz,axis=1)],1)
                zs.append(z)
                activation=np.expand_dims(np.max(a_temp,axis=1),1)
                activations.append(activation)
        """最后一层"""
        z = np.dot(self.weights[-1], activation) + self.biases[-1]
        zs.append(z)
        activation = self.class_activation.fn(z)
        pooling_ind.append(0)
        activations.append(activation)

        delta = (self.cost).delta(zs[-1], activations[-1], y, self.class_activation.prime)
        """最后一层的偏l/偏a"""
        self.Pl_Pa.append(delta*self.class_activation.prime(activations[-1]))
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # insert_zero代表着是否将传递到卷积前一层的w添加0
        insert_zero = False
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activation_fn.prime(z)
            if isinstance(self.sizes[-l],list):
                in_n = self.sizes[-l][0]
            else:
                in_n = self.sizes[-l]
            if insert_zero:
                assert isinstance(self.sizes[-l+1],list)
                w = np.zeros((self.sizes[-l+1][0],in_n))
                for nodes in range(self.sizes[-l+1][0]):
                    w[nodes,pooling_ind[-l+1][nodes]:pooling_ind[-l+1][nodes]+self.sizes[-l+1][1]]=self.weights[-l+1][nodes,:]
                Pl_Px = np.dot(w.T,delta)
            else:
                Pl_Px = np.dot(self.weights[-l+1].transpose(), delta)
            self.Pl_Pa.append(Pl_Px)
            if self.layers_type[-l]=='C':
                delta = Pl_Px*sp
                for nodes in range(len(nabla_w[-l])):
                    nabla_w[-l][nodes,:] = np.dot(delta[nodes],
                                                  (activations[-l-1][pooling_ind[-l][nodes]:pooling_ind[-l][nodes]+self.sizes[-l][-1]]).T)
                insert_zero=True
            else:
                delta = Pl_Px * sp
                nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
                insert_zero=False
            nabla_b[-l] = delta
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        """
        # if convert:
        #     results = [(np.argmax(self.feedforward(x)), np.argmax(y))
        #                for (x, y) in data]
        # else:
        #     results = [(np.argmax(self.feedforward(x)), y)
        #                 for (x, y) in data]
        results = [(self.feedforward(x),y) for (x,y) in data]
        result_accuracy = sum(int(np.where(x>=0.5,1,0) == y) for (x, y) in results)
        if convert:
            return result_accuracy/len(data)
        return result_accuracy

    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)

            cost += self.cost.fn(a, y)/len(data)
            if self.regularizer=='L2':
                cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights) # '**' - to the power of.
            elif self.regularizer=='L1':
                cost += 0.5 * (lmbda / len(data)) * sum(np.linalg.norm(w,ord=1) for w in self.weights)
        return cost

    def get_loss(self,data,lmbda):
        loss = []
        for x, y in data:
            a = self.feedforward(x)
            loss.append(self.cost.fn(a,y))
        return loss


    def predict_pro(self,data):
        pro = []
        for x in data:
            a = self.feedforward(x)
            pro.append(a)
        return  pro

    def predict(self,data):
        z = []
        for x in data:
            a = self.feedforward(x)
            a = 1 if a[:]>=0.5 else 0
            z.append(a)
        return z


    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


#### Miscellaneous functions
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


def relu(z):
    return np.where(z<0, 0, z)


def relu_prime(z):
    return np.where(z<0, 0, 1)

def softmax(z):
    sum = np.sum(np.exp(z))
    return z/sum


