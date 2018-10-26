import numpy as np
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf
from imblearn.over_sampling import SMOTE
class Batches:
    data_train = pd.read_csv("data/train_data.csv", header=None)
    train_labels = pd.read_csv("data/train_labels.csv", header=None)


    data_test = pd.read_csv("data/test_data.csv", header=None)

    number_of_samples = None
    batch_placement = 0
    oh_labels = []

    def __init__(self,oh=True,feature_scaling=True,smote=True):
        self.data_train = self.data_train.values
        self.data_test = self.data_test.values
        self.train_labels = self.train_labels.values
        self.number_of_samples = self.data_train.shape[0]
        if feature_scaling:
            self.feature_scaling()
        if oh:
            self.one_hot()
            self.oh_labels = np.array(self.oh_labels)


    def shuffle(self):
        a = self.data_train
        b = self.train_labels
        c = self.oh_labels
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)

    def get_batch(self,no_batches=100):
        lbls = []
        batches = []
        oh_lbls = []
        lbls = np.array(lbls)
        batches = np.array(batches)
        oh_lbls = np.array(oh_lbls)
        if self.batch_placement + no_batches > self.number_of_samples:
            diff = self.number_of_samples - self.batch_placement
            lbls = (self.train_labels[self.batch_placement:])
            batches = (self.data_train[self.batch_placement:])
            oh_lbls = self.oh_labels[self.batch_placement:]
            self.shuffle()
            lbls = np.append(lbls, [self.train_labels[i] for i in range(no_batches-diff)])
            lbls = np.reshape(lbls,newshape=(no_batches,-1))
            oh_lbls = np.append(oh_lbls,[self.oh_labels[i] for i in range(no_batches-diff)])
            oh_lbls = np.reshape(oh_lbls,newshape=(no_batches,-1))
            batches = np.append(batches, [self.data_train[i] for i in range(no_batches-diff)])
            batches = np.reshape(batches,newshape=(no_batches,self.data_train.shape[1]))
            self.batch_placement = no_batches-diff

        else:
            lbls = (self.train_labels[self.batch_placement:no_batches+self.batch_placement])
            batches = (self.data_train[self.batch_placement:no_batches+self.batch_placement])
            oh_lbls = (self.oh_labels[self.batch_placement:no_batches+self.batch_placement])
            self.batch_placement = self.batch_placement+no_batches


        batches = np.array(batches)

        return oh_lbls, batches

    def one_hot(self,upper_lbl = 10):
        lbls = []
        for lbl in self.train_labels:
            current_lbl = np.zeros(upper_lbl)
            current_lbl[lbl-1] = 1
            lbls.append(current_lbl)
        self.oh_labels = np.array(lbls)

    def train_test_split(self,number_test = 20):
        indexes = []
        indexes_got = []
        import random

        self.train_labels = np.reshape(self.train_labels,(self.train_labels.shape[0]))
        for i in range(1,11):
            indexes.append(np.argwhere(self.train_labels == i))

        for j in range(0,10):
            current_index = indexes[j]
            current_index = np.reshape(current_index,current_index.shape[0])
            indexes_got.append(random.sample(list(current_index),number_test))
        indexes_got = np.array(indexes_got)
        self.data_test_splitted = self.data_train[indexes_got.flatten()]
        self.labels_test_splitted = self.train_labels[indexes_got.flatten()]
        self.oh_test_splitted = self.oh_labels[indexes_got.flatten()]

        self.data_train = np.delete(self.data_train,indexes_got.flatten(),axis=0)
        self.train_labels = np.delete(self.train_labels,indexes_got.flatten(),axis=0)
        self.oh_labels = np.delete(self.oh_labels,indexes_got.flatten(),axis=0)
        self.number_of_samples = self.number_of_samples - number_test*10

    def normal_train_test(self):
        from sklearn.model_selection import train_test_split
        self.data_train, self.data_test_splitted, \
            self.train_labels, self.labels_test_splitted = train_test_split(
            self.data_train,self.train_labels,test_size=0.3)
        self.one_hot()

        lbls = []
        for lbl in self.labels_test_splitted:
            current_lbl = np.zeros(10)
            current_lbl[lbl - 1] = 1
            lbls.append(current_lbl)
        self.oh_test_splitted = np.array(lbls)

        self.number_of_samples = self.data_train.shape[0]
    def feature_scaling(self):
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        self.data_train = min_max_scaler.fit_transform(self.data_train)
        #print(self.data_train.shape)
        #self.data_train = [lambda data : ((data - data.min)/(data.max-data.min)) for data in self.data_train]
        #self.data_train = np.apply_along_axis(lambda : ,0)
        #self.data_train = np.array(self.data_train)
        #print(self.data_train.shape)

    def kaggle_augment(self):
        print(np.shape(self.data_train))
        print(np.shape(self.train_labels))
        means = []
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        self.data_test = min_max_scaler.fit_transform(self.data_test)
        for i in range(1,11):

            current_data = [j for j in range(self.data_train.shape[0]) if self.train_labels[j] == i]
            print(len(current_data))
            current_data_train = self.data_train[current_data]
            print(np.shape(current_data_train))
            means.append(np.mean(current_data_train,axis=0))
        means = np.array(means)
        lbls = []
        print(np.shape(self.data_test))
        for kaggle_vector in self.data_test:
            min_mean = np.linalg.norm(kaggle_vector-means[0])
            mean_lbl = 0
            for i in range(1,10):
                if min_mean < np.linalg.norm(kaggle_vector-means[i]):
                    min_mean = np.linalg.norm(kaggle_vector-means[i])
                    mean_lbl = i
            lbls.append(mean_lbl)
        lbls = np.array(lbls)
        print(lbls.shape)
        a = np.zeros(10)
        for lbl in lbls:
            a[lbl]+=1
        print(a)
        print(np.sum(a))

    def model(self):
        param_nr = 50
        self.data_train = self.data_train[:,:param_nr]
        self.data_test = self.data_test[:,:param_nr]
        self.data_test_splitted = self.data_test_splitted[:,:param_nr]
        y_ = tf.placeholder(tf.float32,[None,10])
        x = tf.placeholder(tf.float32,[None,param_nr])
        initial_W = tf.truncated_normal([param_nr, 10], stddev=0.01)
        W_fc1 = tf.Variable(initial_W)
        initial_b  =  tf.constant(0.0, shape=[10])
        b_fc1 = tf.Variable(initial_b)
        y = tf.matmul(x, W_fc1) + b_fc1
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.04)
        train = optimizer.minimize(cross_entropy)
        saver = tf.train.Saver()

        # initialize the graph
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        for i in range(5000):
            output_, input_  = self.get_batch(100)
            sess.run(train,feed_dict={x:input_,y_:output_})
            if i%100 == 0:
                predict = tf.matmul(np.array(self.data_test_splitted,dtype=np.float32 ),W_fc1)
                predict_result = sess.run(predict)
                sum_ = 0
                for j in range(self.oh_test_splitted.shape[0]):
                    if np.argmax(self.oh_test_splitted[j]) == np.argmax(predict_result[j]):
                        sum_ = sum_ +1
                print("TEST on step "+str(i) + " :" + str(sum_/200))

                predict = tf.matmul(np.array(self.data_train[:4000], dtype=np.float32), W_fc1)
                predict_result = sess.run(predict)
                sum_ = 0
                for j in range(4000):
                    if np.argmax(self.oh_labels[j]) == np.argmax(predict_result[j]):
                        sum_ = sum_ + 1
                print("TRAIN on step " + str(i) + " :" + str(sum_ / 4000))
                rand_nr = np.random.randint(0,4000)
                print(self.oh_labels[rand_nr])
                print(self.train_labels[rand_nr])

    def model_2(self):
        param_nr = 264
        self.data_train = self.data_train[:, :param_nr]
        self.data_test = self.data_test[:, :param_nr]
        self.data_test_splitted = self.data_test_splitted[:, :param_nr]
        y_ = tf.placeholder(tf.float32, [None, 10])
        x = tf.placeholder(tf.float32, [None, param_nr])

        initial_W = tf.truncated_normal([param_nr, 264], stddev=0.01)
        W_1 = tf.Variable(initial_W)
        initial_b_1 = tf.constant(0.0, shape=[264])
        b_1 = tf.Variable(initial_b_1)
        o_1 = tf.matmul(x, W_1) + b_1

        initial_W_2 = tf.truncated_normal([264, 10], stddev=0.01)
        W_2 = tf.Variable(initial_W_2)
        initial_b_2 = tf.constant(0.0, shape=[10])
        b_2 = tf.Variable(initial_b_2)
        y = (tf.matmul(o_1, W_2) + b_2)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.04)
        train = optimizer.minimize(cross_entropy)
        saver = tf.train.Saver()

        # initialize the graph
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        for i in range(50000):
            output_, input_ = self.get_batch(100)
            sess.run(train, feed_dict={x: input_, y_: output_})
            if i % 1000 == 0:
                predict = sess.run(y, feed_dict={x: self.data_test_splitted, y_: self.oh_test_splitted})
                sum_ = 0
                for j in range(self.oh_test_splitted.shape[0]):
                    if np.argmax(self.oh_test_splitted[j]) == np.argmax(predict[j]):
                        sum_ = sum_ + 1
                print("TEST on step " + str(i) + " :" + str(sum_ / 200))

                predict = sess.run(y, feed_dict={x: self.data_train, y_: self.oh_labels})
                sum_ = 0
                for j in range(self.data_train.shape[0]):
                    if np.argmax(self.oh_labels[j]) == np.argmax(predict[j]):
                        sum_ = sum_ + 1
                print("TRAIN on step " + str(i) + " :" + str(sum_ / 4000))
                rand_nr = np.random.randint(0, 4000)
                print(self.oh_labels[rand_nr])
                print(self.train_labels[rand_nr])

    def smote(self):
        sm = SMOTE(random_state=2)
        self.data_train,self.train_labels = sm.fit_sample(self.data_train,self.train_labels.ravel())
        self.number_of_samples = self.data_train.shape[0]


# b = Batches(smote=False)
# b.kaggle_augment()

# a = Batches()
# a.one_hot()
# a.train_test_split()
# a.feature_scaling()
# print("Before:")
# print("Data_train:" + str(a.data_train.shape))
# print("Label_test:" +str(a.train_labels.shape))
# print("OH:"+str(a.oh_labels.shape))
# a.smote()
# a.one_hot()
# print("After:")
# print("Data_train:" + str(a.data_train.shape))
# print("Label_test:" +str(a.train_labels.shape))
# print("OH:"+str(a.oh_labels.shape))
#
# print("No of samples: "+str(a.number_of_samples))
# a = Batches()
# z = [1,2,3]
# b = [1,2,3]
# a.one_hot()
# a.feature_scaling()
# a.train_test_split()
#
#
# print("oh_labels")
# print(np.shape(a.oh_labels))
# print("train_labels")
# print(np.shape(a.train_labels))
# print("data train")
# print(np.shape(a.data_train))

# print("oh_test")
# print(np.shape(a.oh_test_splitted))
# print("train_labels")
# print(np.shape(a.labels_test_splitted))
# print("data test")
# print(np.shape(a.data_test_splitted))

# a.model_2()


#for i in range(100):
#    print("Iter no " + str(i))
#    y, x = a.get_batch(1000)
#    print(y.shape)
#    print(x.shape)
#    print(a.batch_placement)
#    print("\n")

# a.shuffle()