# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 21:26:58 2019

@author: Hasan
"""

# -- coding: utf-8 --
"""
Created on Fri Mar  1 21:25:49 2019

@author: Hasan
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("data/Mnist/",one_hot=True)

x=tf.placeholder(tf.float32,[None,784])
y_true=tf.placeholder(tf.float32,[None,10])

layer_1=128
layer_2=64
layer_3=32
layer_out=10

weigth_1=tf.Variable(tf.truncated_normal([784,layer_1],stddev=0.1))# rasgale değerler atıyoruz ve standart sapma vererek bir birlerine yakın değerler atanıyor
bais_1=tf.Variable(tf.constant(0.1,shape=[layer_1]))

weigth_2=tf.Variable(tf.truncated_normal([layer_1,layer_2],stddev=0.1))
bais_2=tf.Variable(tf.constant(0.1,shape=[layer_2]))

weigth_3=tf.Variable(tf.truncated_normal([layer_2,layer_3],stddev=0.1))
bais_3=tf.Variable(tf.constant(0.1,shape=[layer_3]))

weigth_4=tf.Variable(tf.truncated_normal([layer_3,layer_out],stddev=0.1))
bais_4=tf.Variable(tf.constant(0.1,shape=[layer_out]))

y1=tf.nn.relu(tf.matmul(x,weigth_1)+bais_1)
y2=tf.nn.relu(tf.matmul(y1,weigth_2)+bais_2)
y3=tf.nn.relu(tf.matmul(y2,weigth_3)+bais_3)
logits=tf.matmul(y3,weigth_4)+bais_4
y4=tf.nn.softmax(logits)

xend=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_true)
loss=tf.reduce_mean(xend)

correct_prediction=tf.equal(tf.argmax(y4,1),tf.argmax(y_true,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

optimize=tf.train.AdamOptimizer(0.001).minimize(loss)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

batch_size=128
loss_graph=[]
def traning_step(iteration):
    for i in range(iteration):
        x_batch,y_batch=mnist.train.next_batch(batch_size)
        feed_dict_train={x:x_batch ,y_true:y_batch}
        [_,train_loss]=sess.run([optimize,loss],feed_dict=feed_dict_train)
        loss_graph.append(train_loss)
        if i%100==0:
            train_acc=sess.run(accuracy,feed_dict=feed_dict_train)
            print("iterations : ",i," Trainnig accuracy : ",train_acc," Tranning loss:",train_loss)
            
        
feed_dict_test={x:mnist.test.images ,y_true:mnist.test.labels}
def test_accuracy(f_d_t):
    #feed_dict_test={x:mnist.test.images ,y_true:mnist.test.labels,pkeep:1}
    acc=sess.run(accuracy,feed_dict=f_d_t)
    print('Test oran :',acc)
    
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(28, 28), cmap='binary')
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def plot_example_errors():
    mnist.test.cls = np.argmax(mnist.test.labels, axis=1)
    y_pred_cls = tf.argmax(y4, 1)
    correct, cls_pred = sess.run([correct_prediction, y_pred_cls], feed_dict=feed_dict_test)
    incorrect = (correct == False)

    images = mnist.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = mnist.test.cls[incorrect]

    plot_images(images=images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])

traning_step(10000)
test_accuracy(feed_dict_test)

plot_example_errors()
#plot_images()

plt.plot(loss_graph,'k-')
plt.title('Loss Grafiği')
plt.xlabel('İterasyon')
plt.ylabel('Loss')
plt.show()