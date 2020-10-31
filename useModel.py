import tensorflow as tf
import argparse as aps
from tensorflow.contrib import data
from heapq import nlargest
import os

IMAGE_WIDTH = 108
IMAGE_HEIGHT = 192
N_CLASSES = 2

parser = aps.ArgumentParser(description="manual to this script")
parser.add_argument("--model",type=str,default="model/mod.ckpt-2000")
args = parser.parse_args()

def fileName(filename):
    for _, dirs, files in os.walk(filename):
        return files

test = fileName("data/testdata/")
# 模型地址
MODEL_PATH = args.model
# 读取图像
def read_image_tensor(image_dir):
    image = tf.gfile.FastGFile(image_dir, 'rb').read()
    image = tf.image.decode_jpeg(image) #图像解码
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8) #改变图像数据的类型
    image = tf.image.resize_images(image, [IMAGE_WIDTH, IMAGE_HEIGHT], method=0)
    image = tf.image.per_image_standardization(image)
    image = tf.reshape(image, [1, IMAGE_WIDTH, IMAGE_HEIGHT, 3])
    return image

def useModel():
    # 分类标签
    labels = ['拍照', '非拍照']
    model = tf.train.import_meta_graph(MODEL_PATH+".meta")
    graph = tf.get_default_graph()
    inputs = graph.get_operation_by_name('x-input').outputs[0]
    is_train = graph.get_operation_by_name('is_train').outputs[0]
    pred = tf.get_collection('pred_network')[0]
    test_list = {}
    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        model.restore(sess, MODEL_PATH)
        for path in test:
            image = read_image_tensor("data/testdata/"+path)
            image = sess.run(image)
            pred_y = sess.run(tf.nn.softmax(pred,1), feed_dict={inputs:image,is_train:False})
            max_index = list(map(list(pred_y[0]).index,nlargest(4,pred_y[0])))
            max_num = nlargest(4,pred_y[0])
            print(path)
            print("预测类别前三 ： ")
            test_list[path] = labels[max_index[0] - 1]
            print("\t",labels[max_index[0] - 1],":",max_num[0]*100,"%")
            # print("\t",labels[max_index[1] - 1],":",max_num[1]*100,"%")
            # print("\t",labels[max_index[2] - 1],":",max_num[2]*100,"%")


#传入模型
if __name__ == '__main__':
    with tf.device("/cpu:0"):
        useModel()
