# coding: utf-8
#!/usr/bin/env python

import os 
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import io, transform
import skimage
from tensorflow_vgg import vgg16
from tensorflow_vgg import utils
from tensorflow.python.platform import gfile
import base64
from io import BytesIO
import time

from flask import Flask,render_template,request
app = Flask(__name__)


def load_image(path):
    """根据图片路径读取图片，先进行居中切割，再转为float型，后缩放为224"""
    # load image
    img = skimage.io.imread(path)
#     img = img / 255.0
#     assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(img, (224, 224))
    return resized_img


def get_binstr_by_resized_img(binstr, cache_path='./cache_pic/', ):
    """解码binstr得到图片，并将其本地缓存，再读取该数据, 返回为数组"""
    
    img = Image.open(BytesIO(base64.b64decode(binstr)))
    
    local_time = time.strftime("%Y-%m-%d  %H_%M_%S",time.localtime(time.time()))
    save_path = cache_path + local_time + '.jpg'
    img.save(save_path, 'JPEG')
    
    resized_img = load_image(save_path)
    
    return resized_img, save_path

vege_dict = {0:'jmc', 1:'qc', 2:'qingcai'}
pics_path_list = {}



def decode_and_copy_pic(pic_b64str):
    pic_path = get_binstr_by_resized_img(binstr)[1]
    pics_path_list.append(pic_path)
    

def get_vegelabel_from_b64str(b64str):
    with tf.Graph().as_default() as g:
        with tf.Session() as sess:
            vgg = vgg16.Vgg16()
            input_test = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='input_test')
            inputs_fea = tf.placeholder(tf.float32, shape=[None , 4096], name='inputs_fea')

            with tf.name_scope("content_vgg"):
                # 先准备数据
                
                batch_list = []
                image, image_path = get_binstr_by_resized_img(b64str)
                batch_list.append(image.reshape((1, 224, 224, 3)))
                test_images = np.concatenate(batch_list, 0)
                
                vgg.build(input_test)
                feature_codes = sess.run(vgg.relu6, feed_dict={input_test:test_images})
                print (feature_codes)
        #             print ("feature_codes.shape" , feature_codes.shape)
                assert feature_codes.shape == (2, 4096)


            # 加入一个256维的全连接的层
            fc = tf.contrib.layers.fully_connected(inputs_fea, 256)
            logits = tf.contrib.layers.fully_connected(fc, 3, activation_fn=None)
            predicted = tf.nn.softmax(logits, name='predicted')


            ###  创建saver对象时，必须要定义图结构，如上面的fc logits  predicted 
            saver = tf.train.Saver()
            saver.restore(sess, "checkpoints/zsy.ckpt")  # 注意这里只恢复变量值
            ## 需指定要恢复的操作符或张量， 读取到对应的graph对象中

    #         g = tf.get_default_graph()
    #         g.get_tensor_by_name('predicted:0')
            prob_op = g.get_operation_by_name('predicted')
            pred_result = sess.run(predicted, feed_dict={inputs_fea : feature_codes})
            print (pred_result)
            result_list = tf.argmax(pred_result , 1 ).eval()
            print (result_list)
            print (type(result_list))
            for v in result_list:
                return (vege_dict[v])
    




@app.route('/')
def index():
    return 'Index Page'

@app.route('/hello')
def hello_world():
    return "Hello World!"


@app.route('/recognize_vege', methods=['POST'])
def register():
    pic_b64str = request.form['pic_str']
    
    vege_label = get_vegelabel_from_b64str(pic_b64str)
    print (vege_label)
    
    return vege_label

if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)




