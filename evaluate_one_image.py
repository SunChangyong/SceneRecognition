# -*- coding:utf-8 -*-
"""
Created on Mon Jul 24 05:33:09 2017

@author: hjxu
"""

#去掉不必要的warning　message
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#%% Evaluate one image
# when training, comment the following codes.
import model
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# import create_records

# def get_one_image(train):
#     '''Randomly pick one image from training data
#     Return: ndarray
#     '''
#     n = len(train)
#     ind = np.random.randint(0, n)
#     img_dir = train[ind]
#  
#     image = Image.open(img_dir)
#     plt.imshow(image)
#     image = image.resize([208, 208])
#     image = np.array(image)
#     return image

scene = ['diningroom','livingroom','bathroom','stairs','bedroom']

def get_one_img(img_dir):
    image = Image.open(img_dir)
    
    img = image
    img.resize((512,512),Image.ANTIALIAS).show() 
    
    plt.imshow(image)
    image = image.resize([208, 208])
    image = np.array(image)
    return image
    
def evaluate_one_image():
    '''Test one image against the saved models and parameters
    '''
    
    # you need to change the directories to yours.
    img_dir = '/home/scy/eclipse-workspace/PythonProject/src/12yue3ri/test_images/022.jpg'
    image_array = get_one_img(img_dir)
#     train_dir = '/home/scy/eclipse-workspace/PythonProject/src/test_own_dataset/train_images/'
#     train, train_label = create_records.get_files(train_dir)
#     image_array = get_one_image(img_dir)

    
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 5
        
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 208, 208, 3])
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)
        
        logit = tf.nn.softmax(logit)
        
        x = tf.placeholder(tf.float32, shape=[208, 208, 3])
        
        # you need to change the directories to yours.
        logs_train_dir = '/home/scy/eclipse-workspace/PythonProject/src/12yue3ri/logs/' 
                       
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            
            print("Reading checkpoints...")
            
            saver.restore(sess,'net/mydataset_net.ckpt')
            print('Loading succeed.')
            
#             ckpt = tf.train.get_checkpoint_state(logs_train_dir)            
#             if ckpt and ckpt.model_checkpoint_path:
#                 global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#                 saver.restore(sess, ckpt.model_checkpoint_path)
#                 print('Loading success, global_step is %s' % global_step)
#             else:
#                 print('No checkpoint file found')
            
            prediction = sess.run(logit, feed_dict={x: image_array})
            print(prediction)
            
            max_index = np.argmax(prediction)
            print('This is a %s with possibility %.6f' % (scene[max_index], prediction[:, max_index]))
            
            print('\nAll possibilities:')
            for i in range(5):
                print('This is a %s with possibility %.6f' % (scene[i], prediction[:, i]))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    evaluate_one_image()
