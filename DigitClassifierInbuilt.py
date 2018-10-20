# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 00:00:09 2018

@author: user
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


mnist = input_data.read_data_sets("mnist_data",one_hot=True)


observation = len(mnist.train.images)
features = len(mnist.train.images[1])

print ("\n Size of dataset is with {:d} observation and {:d} featurs".format(observation,features))


#--------- Define global variables.
batch_size = 100        #-- Batch size.Number of records feed to algorith in on go.
epochs = 15             #-- Number of times we repeat learning process.
learning_rate = 0.01    #-- Rate of learning by algorith
hidden_layer_1= 256     #-- Hidder layer1 with 256 neurons
hidden_layer_2= 256     #-- Hidder layer2 with 256 neurons
input_layer= 784        #-- Inpurt layer withh 784 ( 28*28) image
output = 10             #-- 10 possilility for 10 digits. Used as one hot.
observations = mnist.train.num_examples
batches = int(observations / batch_size)

#--------- Create place holders to hold the inpurt data and output label

x  = tf.placeholder (tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])
    
weights ={
            'theta1':tf.Variable(tf.random_normal([784,256])),
            'theta2':tf.Variable(tf.random_normal([256,256])),
            'theta3':tf.Variable(tf.random_normal([256,10]))
            
        }
biases = {
            'b1': tf.Variable(tf.random_normal([256])),
            'b2': tf.Variable(tf.random_normal([256])),
            'b3': tf.Variable(tf.random_normal([10])),
        
        }

#--------- Define algorithm for Digit Classifier

hid_layer_1 = tf.add(tf.matmul(x,weights['theta1']),biases['b1'])
hid_layer_1 = tf.nn.relu(hid_layer_1)
hid_layer_2 = tf.add(tf.matmul(hid_layer_1,weights['theta2']),biases['b2'])
hid_layer_2 = tf.nn.relu(hid_layer_2)
output_layer = tf.matmul(hid_layer_2,weights['theta3'])+biases['b3']

#--------- For final output layer. We will use softmax and use this as a cost function
#-- To improve learning of algorithm
pred = tf.nn.softmax_cross_entropy_with_logits(logits=output_layer,labels=y_)
#-- Cost of algorithm
cost = tf.reduce_mean(pred)

#-- Define optimizer to optimize cost.We want to minimize the cost, so we have provided cost function to th 
#-- AdamOptimizer.
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
#
#


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    accuracy_history=[]
    cost_history=[]
    print("\n----------------- Learning data ------------------------- ")
    for epoc in range(epochs):
        avg_cost = 0
        for _ in range(batches):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            _,c = sess.run([optimizer,cost],feed_dict={x:x_batch,y_:y_batch})
            avg_cost += c/batches
#            print(sess.run(output_layer,feed_dict={x:x_batch}))
        prediction = tf.arg_max(output_layer,1)
        #print(prediction)
        actual = tf.arg_max(y_,1)
        correct_pred = tf.equal(prediction,actual)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        cost_history.append(avg_cost)
        accu_train = accuracy.eval({x:mnist.train.images,y_:mnist.train.labels})
        accu_test = accuracy.eval({x:mnist.test.images,y_:mnist.test.labels})
        
#        print("Accuracy on test data set is {:f} and Accuracy on train dataset is {:f}".format(accu_test,accu_train ))
    
    print("\n----------------- Learning Completed ------------------------- ")
    print("\n Evaluating Algorith After Training")
    trained_accuracy = accuracy.eval({x:mnist.test.images,y_:mnist.test.labels})
    print("\n--- Accuracy after training data is {:f}".format(trained_accuracy))
    output = prediction.eval({x:mnist.test.images,y_:mnist.test.labels})
    print(output[1])
    print(tf.argmax(mnist.test.labels,1).eval()[1])
            
    
#    for epoc in range(epochs):
#        
#        for _ in range(batches):
#            
#            x_batch, y_batch = mnist.train.next_batch(batch_size)
#            print(sess.run(optimizer,feed_dict={x:x_batch,y:y_batch}))
            
    
    