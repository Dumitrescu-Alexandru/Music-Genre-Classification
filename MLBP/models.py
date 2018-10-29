import pandas as pd
from MLBP.batches import Batches
from sklearn import preprocessing
import tensorflow as tf
import numpy as np
import datetime
import csv


class Model():

    data = Batches()

    def __init__(self,noise=True,final=False,smote=True,normal_split=True):
        self.data.one_hot()
        self.data.feature_scaling()


        if not final and normal_split:
            self.data.normal_train_test()
            print(self.data.data_train.shape)
            print(self.data.train_labels.shape)
            print(self.data.oh_labels.shape)


            print(self.data.data_test_splitted.shape)
            print(self.data.labels_test_splitted.shape)
            print(self.data.oh_test_splitted.shape)
        elif not final and not normal_split:
            self.data.train_test_split(number_test=10)
            print(self.data.data_train.shape)
            print(self.data.data_test_splitted.shape)
        if smote:
            self.data.smote()
        self.data.one_hot()

    def L3_NN_model(self,augment=True,learning_rate=0.01,no_batches=200,save_model=True,no_layers=1,dims=None,folder='results',
                    no_features=264,keep_prob=0.8,save_results=True,predictor_name="y",print_test=True,no_steps=2000,
                    confusion_matrix = False):
        tf.reset_default_graph()
        if dims == None:
            dims = np.ones([no_layers,2])
            for j in range(no_layers):
                dims[j,0] = no_features
                dims[j,1] = no_features
            dims[-1,-1] = 10
        dims = np.array(dims,dtype=np.int32)
        y_ = tf.placeholder(tf.float32, [None, 10])
        x = tf.placeholder(tf.float32, [None, no_features])
        self.data.data_train= self.data.data_train[:,:no_features]

        if print_test:
            self.data.data_test_splitted = self.data.data_test_splitted[:,:no_features]
        iW_1 = tf.random_normal([dims[0,0],dims[0,1]],stddev=0.1)
        iW_2 = tf.random_normal([dims[1,0],dims[1,1]],stddev=0.1)
        iW_3 = tf.random_normal([dims[2,0],dims[2,1]],stddev=0.1)

        ib_1 = tf.random_normal([dims[0,1]],stddev=0.1)
        ib_2 = tf.random_normal([dims[1,1]],stddev=0.1)
        ib_3 = tf.random_normal([dims[2,1]],stddev=0.1)
        W_1 = tf.Variable(iW_1,name="W0")
        W_2 = tf.Variable(iW_2,name="W1")
        W_3 = tf.Variable(iW_3,name="W2")

        b_1 = tf.Variable(ib_1,name="b0")
        b_2 = tf.Variable(ib_2,name="b1")
        b_3 = tf.Variable(ib_3,name="b2")
        y_1 = tf.nn.sigmoid(tf.matmul(x,W_1)+b_1)
        y_1_dropped = tf.nn.dropout(y_1,keep_prob=keep_prob)
        y_2 = tf.nn.sigmoid(tf.matmul(y_1_dropped,W_2)+b_2)
        y_2_dropped = tf.nn.dropout(y_2,keep_prob=keep_prob)
        y_3 = tf.matmul(y_2_dropped,W_3)+b_3
        y = y_3
        norms = [2178,618, 326, 253, 214, 260, 141,195, 92, 86]

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train = optimizer.minimize(cross_entropy)
        file_name = "LAYERS_" + str(no_layers) + "_KeepProb_" + str(keep_prob) + "_Batches_" + str(no_batches) + \
                    "_LrnRate_" + str(learning_rate)

        if save_results:

            file_write_test = open(folder + "/TEST_" + file_name, "wt")
            file_write_train = open(folder + "/TRAIN_" + file_name, "wt")
            writer_test = csv.writer(file_write_test, delimiter=',')
            writer_train = csv.writer(file_write_train, delimiter=',')



        saver = tf.train.Saver()
        init = tf.initialize_all_variables()
        sess = tf.Session()
        if augment:
            tpot_pred = pd.read_csv("results_kaggle/tpot_accuracy_solution.csv")
            tpot_vals = tpot_pred.values
            tpot_vals = tpot_vals[:, 1]
            lbls_oh = []
            for a in tpot_vals:
                current_lbl = np.zeros(10)
                current_lbl[a - 1] = 1
                lbls_oh.append(current_lbl)
            self.data_augment_lbls = np.array(lbls_oh)
        sess.run(init)
        for i in range(no_steps):
            output_, input_ = self.data.get_batch(no_batches)
            if augment:
                indexes = np.random.randint(0,6544,size=(20))
                sampl_features = self.data.data_test[indexes]
                sampl_lbls = self.data_augment_lbls[indexes]
                output_ = np.concatenate((output_,sampl_lbls))
                input_ = np.concatenate((input_,sampl_features))

            sess.run(train, feed_dict={x: input_, y_: output_})
            if i % 100 == 0:

                if print_test:
                    predict = sess.run(y, feed_dict={x: self.data.data_test_splitted, y_: self.data.oh_test_splitted})
                    sum_ = 0
                    for j in range(self.data.oh_test_splitted.shape[0]):
                        if np.argmax(self.data.oh_test_splitted[j]) == np.argmax(predict[j]):
                            sum_ = sum_ + 1
                    print("TEST on step " + str(i) + " :" + str(sum_ / self.data.oh_test_splitted.shape[0]))
                    if save_results:
                        writer_test.writerow([i,sum_/self.data.oh_test_splitted.shape[0]])

                predict = sess.run(y,feed_dict={x:self.data.data_train,y_:self.data.oh_labels})
                sum_ = 0
                for j in range(self.data.data_train.shape[0]):
                    if np.argmax(self.data.oh_labels[j]) == np.argmax(predict[j]):
                        sum_ = sum_ + 1
                print("TRAIN on step " + str(i) + " :" + str(sum_/self.data.data_train.shape[0]))
                if save_results:
                    writer_train.writerow([i,sum_/self.data.data_train.shape[0]])
        if save_model:
            saver.save(sess,  folder + "/"+ file_name)
        if confusion_matrix:
            predict_conf = sess.run(y,feed_dict={x:self.data.data_test_splitted})
            print(predict_conf[:20])
            print(self.data.labels_test_splitted)
            conf_matrix = np.zeros((10,10))
            for iter in range(predict_conf.shape[0]):
                conf_matrix[np.argmax(self.data.oh_test_splitted[iter]),
                            np.argmax(predict_conf[iter])] += 1
            print(conf_matrix)
            df = pd.DataFrame(conf_matrix, index=None)
            print(df)
            df.to_csv("template_results/Confusion_matrix_" + file_name + ".csv")
    def NN_model_2L(self,augment=True,learning_rate=0.01,no_batches=200,save_model=True,no_layers=1,dims=None,no_features=264,keep_prob=0.8,
                    save_results=True,predictor_name="y",print_test=True,no_steps=2000,folder='results',confusion_matrix=True):
        if dims == None:
            dims = np.ones([no_layers,2])
            for j in range(no_layers):
                dims[j,0] = no_features
                dims[j,1] = no_features
            dims[-1,-1] = 10
        dims = np.array(dims,dtype=np.int32)
        y_ = tf.placeholder(tf.float32, [None, 10])
        x = tf.placeholder(tf.float32, [None, no_features])
        self.data.data_train= self.data.data_train[:,:no_features]
        if print_test:
            self.data.data_test_splitted = self.data.data_test_splitted[:,:no_features]
        iW_1 = tf.random_normal([dims[0,0],dims[0,1]],stddev=0.1)
        iW_2 = tf.random_normal([dims[1,0],dims[1,1]],stddev=0.1)

        ib_1 = tf.random_normal([dims[0,1]],stddev=0.1)
        ib_2 = tf.random_normal([dims[1,1]],stddev=0.1)
        W_1 = tf.Variable(iW_1,name="W0")
        W_2 = tf.Variable(iW_2,name="W1")

        b_1 = tf.Variable(ib_1,name="b0")
        b_2 = tf.Variable(ib_2,name="b1")
        y_1 = tf.nn.sigmoid(tf.matmul(x,W_1)+b_1)
        y_1_dropped = tf.nn.dropout(y_1,keep_prob=keep_prob)
        y_2 = tf.matmul(y_1_dropped,W_2)+b_2
        y=y_2
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train = optimizer.minimize(cross_entropy)
        file_name = "LAYERS_" + str(no_layers) + "_KeepProb_" + str(keep_prob) + "_Batches_" + str(no_batches) + \
                    "_LrnRate_" + str(learning_rate)

        if save_results:

            file_write_test = open(folder + "/TEST_" + file_name, "wt")
            file_write_train = open(folder + "/TRAIN_" + file_name, "wt")
            writer_test = csv.writer(file_write_test, delimiter=',')
            writer_train = csv.writer(file_write_train, delimiter=',')



        saver = tf.train.Saver()
        init = tf.initialize_all_variables()
        sess = tf.Session()
        if augment:
            tpot_pred = pd.read_csv("results_kaggle/tpot_accuracy_solution.csv")
            tpot_vals = tpot_pred.values
            tpot_vals = tpot_vals[:, 1]
            lbls_oh = []
            for a in tpot_vals:
                current_lbl = np.zeros(10)
                current_lbl[a - 1] = 1
                lbls_oh.append(current_lbl)
            self.data_augment_lbls = np.array(lbls_oh)
        sess.run(init)
        for i in range(no_steps):
            output_, input_ = self.data.get_batch(no_batches)
            if augment:
                indexes = np.random.randint(0,6544,size=(100))
                sampl_features = self.data.data_test[indexes]
                sampl_lbls = self.data_augment_lbls[indexes]
                output_ = np.concatenate((output_,sampl_lbls))
                input_ = np.concatenate((input_,sampl_features))

            sess.run(train, feed_dict={x: input_, y_: output_})
            if i % 100 == 0:

                if print_test:
                    predict = sess.run(y, feed_dict={x: self.data.data_test_splitted, y_: self.data.oh_test_splitted})
                    sum_ = 0
                    for j in range(self.data.oh_test_splitted.shape[0]):
                        if np.argmax(self.data.oh_test_splitted[j]) == np.argmax(predict[j]):
                            sum_ = sum_ + 1
                    print("TEST on step " + str(i) + " :" + str(sum_ / self.data.oh_test_splitted.shape[0]))
                    if save_results:
                        writer_test.writerow([i,sum_/self.data.oh_test_splitted.shape[0]])

                predict = sess.run(y,feed_dict={x:self.data.data_train,y_:self.data.oh_labels})
                sum_ = 0
                for j in range(self.data.data_train.shape[0]):
                    if np.argmax(self.data.oh_labels[j]) == np.argmax(predict[j]):
                        sum_ = sum_ + 1
                print("TRAIN on step " + str(i) + " :" + str(sum_/self.data.data_train.shape[0]))
                if save_results:
                    writer_train.writerow([i,sum_/self.data.data_train.shape[0]])
        if save_model:
            saver.save(sess, folder + "/"+file_name)

        if confusion_matrix:
            predict_conf = sess.run(y,feed_dict={x:self.data.data_test_splitted})
            print(predict_conf[:20])
            print(self.data.labels_test_splitted)
            conf_matrix = np.zeros((10,10))
            for iter in range(predict_conf.shape[0]):
                conf_matrix[np.argmax(self.data.oh_test_splitted[iter]),
                            np.argmax(predict_conf[iter])] += 1
            print(conf_matrix)
            df = pd.DataFrame(conf_matrix,index=None)
            print(df)
            df.to_csv("template_results/Confusion_matrix_"+file_name + ".csv",sep=',')

    def load_model(self, file_name, accuracy_kaggle=True,log_loss_kaggle=True, no_layers=1,no_features=264,dims=None,sanity_check=True):
        with tf.Session() as sess:


            x = tf.placeholder(tf.float32, [None, no_features])
            save = tf.train.import_meta_graph(file_name)
            save.restore(sess, tf.train.latest_checkpoint('final_models/'))
            graph = tf.get_default_graph()


            W_0 = graph.get_tensor_by_name("W0:0")
            W_1 = graph.get_tensor_by_name("W1:0")
            b_0 = graph.get_tensor_by_name("b0:0")
            b_1 = graph.get_tensor_by_name("b1:0")
            y_1 = tf.nn.sigmoid(tf.matmul(x,W_0)+b_0)
            y   = tf.nn.sigmoid(tf.matmul(y_1,W_1)+b_1)



            if sanity_check:
                sum_ = 0
                predict = sess.run(y, feed_dict={x: self.data.data_train})
                for j in range(self.data.data_train.shape[0]):
                    if np.argmax(self.data.oh_labels[j]) == np.argmax(predict[j]):
                        sum_ = sum_ + 1
                print("TRAIN  :" + str(sum_ / self.data.data_train.shape[0]))

            import pandas as pd
            data_test = pd.read_csv("data/test_data.csv", header=None)
            data_test = data_test.values
            min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            data_test = min_max_scaler.fit_transform(data_test)
            if accuracy_kaggle:
                predict = sess.run(y,feed_dict={x:data_test})
                predicted = np.zeros(predict.shape[0])

                file_name = datetime.datetime.now().strftime("%I_%M%p_on_%d_%B_%Y")
                accuracy_kaggle_file = open("results_kaggle/Kaggle_accuracy_" + file_name+".csv", "wt",newline='')
                writer_test = csv.writer(accuracy_kaggle_file, delimiter=',')
                writer_test.writerow(["Sample_id","Sample_label"])
                for j in range(predict.shape[0]):
                    predicted[j] = np.argmax(predict[j])
                    writer_test.writerow([j+1,predicted[j]+1])

            if log_loss_kaggle:
                predict = sess.run(y,feed_dict={x:data_test})
                file_name = datetime.datetime.now().strftime("%I_%M%p_on_%d_%B_%Y")
                accuracy_kaggle_file = open("results_kaggle/Kaggle_logloss_" + file_name + ".csv", "wt", newline='')
                writer_test = csv.writer(accuracy_kaggle_file, delimiter=',')
                writer_test.writerow(["Sample_id", "Class_1","Class_2","Class_3","Class_4","Class_5",
                                      "Class_6","Class_7","Class_8","Class_9","Class_10"])
                for j in range(predict.shape[0]):
                    probs = predict[j]/np.sum(predict[j])
                    row = np.concatenate(([j+1], probs))
                    writer_test.writerow(row)

    def load_model_3L(self, file_name, accuracy_kaggle=True,log_loss_kaggle=True, no_layers=1,
                      no_features=264,dims=None,sanity_check=True,confusion_matrix=True):
        with tf.Session() as sess:


            x = tf.placeholder(tf.float32, [None, no_features])



            #tf.reset_default_graph()
            W_0 = tf.get_variable("W0",shape=[264,264*2])
            W_1 = tf.get_variable("W1",shape=[264*2,264])
            W_2 = tf.get_variable("W2",shape=[264,10])
            b_0 = tf.get_variable("b0",shape=[264*2])
            b_1 = tf.get_variable("b1",shape=[264])
            b_2 = tf.get_variable("b2",shape=[10])

            y_1 = tf.nn.sigmoid(tf.matmul(x, W_0) + b_0)
            y_2 = tf.nn.sigmoid(tf.matmul(y_1,W_1)+b_1)
            y = tf.nn.sigmoid(tf.matmul(y_2,W_2)+b_2)

            saver =tf.train.Saver()
            with tf.Session() as sess:
                saver.restore(sess, file_name)
                if sanity_check:
                    sum_ = 0
                    predict = sess.run(y, feed_dict={x: self.data.data_train})
                    for j in range(self.data.data_train.shape[0]):
                        if np.argmax(self.data.oh_labels[j]) == np.argmax(predict[j]):
                            sum_ = sum_ + 1
                    print("TRAIN  :" + str(sum_ / self.data.data_train.shape[0]))

                import pandas as pd
                data_test = pd.read_csv("data/test_data.csv", header=None)
                data_test = data_test.values
                min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
                data_test = min_max_scaler.fit_transform(data_test)
                if accuracy_kaggle:
                    predict = sess.run(y,feed_dict={x:data_test})
                    predicted = np.zeros(predict.shape[0])

                    file_name = datetime.datetime.now().strftime("%I_%M%p_on_%d_%B_%Y")
                    accuracy_kaggle_file = open("results_kaggle/Kaggle_accuracy_" + file_name+".csv", "wt",newline='')
                    writer_test = csv.writer(accuracy_kaggle_file, delimiter=',')
                    writer_test.writerow(["Sample_id","Sample_label"])
                    for j in range(predict.shape[0]):
                        predicted[j] = np.argmax(predict[j])
                        writer_test.writerow([j+1,predicted[j]+1])

                if log_loss_kaggle:
                    predict = sess.run(y,feed_dict={x:data_test})
                    file_name = datetime.datetime.now().strftime("%I_%M%p_on_%d_%B_%Y")
                    accuracy_kaggle_file = open("results_kaggle/Kaggle_logloss_" + file_name + ".csv", "wt", newline='')
                    writer_test = csv.writer(accuracy_kaggle_file, delimiter=',')
                    writer_test.writerow(["Sample_id", "Class_1","Class_2","Class_3","Class_4","Class_5",
                                          "Class_6","Class_7","Class_8","Class_9","Class_10"])
                    for j in range(predict.shape[0]):
                        probs = predict[j]/np.sum(predict[j])
                        row = np.concatenate(([j+1], probs))
                        writer_test.writerow(row)

    def print_no_parameters(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            print(shape)
            variable_parameters = 1
            for dim in shape:
                print(dim)
                variable_parameters *= dim.value
            print(variable_parameters)
            total_parameters += variable_parameters
        print(total_parameters)



model = Model(final=False,smote=False,normal_split=True)
parameters = 264
learning_rates = [0.001,0.003,0.005,0.01,0.03]
keep_probs = [0.7,0.8,0.85,0.9,0.95]

# for lr in learning_rates:
#     for kp in keep_probs:
#         model.NN_model_3L(print_test=True,learning_rate=lr,no_batches=100,
#                           no_layers=3,dims=[[264,264*2],[2*264,264],[264,10]],keep_prob=kp,augment=False)

#tf.reset_default_graph()
model.NN_model_2L(print_test=True,learning_rate=0.003,no_batches=100,
                 no_layers=2,dims=[[264,264],[264,10]],keep_prob=0.85,
                 augment=False,no_steps=2000,folder='template_results',save_model=False,confusion_matrix=True)

# model.L3_NN_model(print_test=True,learning_rate=0.001,no_batches=100,
#                  no_layers=3,dims=[[264,264*2],[264*2,264],[264,10]],keep_prob=0.95,
#                 augment=False,no_steps=3500,folder='template_results',
#                   save_results=False,confusion_matrix=True)
#model.load_model("template_results/LAYERS_2_KeepProb_0.85_Batches_100_LrnRate_0.003.meta",no_layers=3)
# model.load_model("template_results/LAYERS_2_KeepProb_0.85_Batches_100_LrnRate_0.003.meta",no_layers=2)
