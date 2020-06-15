import numpy as np
from detection.MtcnnDetector import MtcnnDetector
from detection.detector import Detector
from detection.fcn_detector import FcnDetector
from networks.mtcnn_model import P_Net, R_Net, O_Net
import cv2
import os
import sklearn
from utils import face_preprocess
import tensorflow as tf
import config


# 比较人脸相似度
def feature_compare(feature1, feature2, threshold):
    dist = np.sum(np.square(feature1 - feature2))
    sim = np.dot(feature1, feature2.T)
    if sim > threshold:
        return True, sim
    else:
        return False, sim


# 加载人脸检测模型
def load_mtcnn():
    MODEL_PATH = config.MTCNN_MODEL_PATH
    MIN_FACE_SIZE = int(config.MIN_FACE_SIZE)
    STEPS_THRESHOLD = [float(i) for i in config.STEPS_THRESHOLD.split(",")]

    detectors = [None, None, None]
    prefix = [MODEL_PATH + "/PNet_landmark/PNet",
              MODEL_PATH + "/RNet_landmark/RNet",
              MODEL_PATH + "/ONet_landmark/ONet"]
    epoch = [18, 14, 16]
    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
    PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet
    RNet = Detector(R_Net, 24, 1, model_path[1])
    detectors[1] = RNet
    ONet = Detector(O_Net, 48, 1, model_path[2])
    detectors[2] = ONet
    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=MIN_FACE_SIZE, threshold=STEPS_THRESHOLD)

    return mtcnn_detector


# 加载已经注册的人脸
def load_faces(sess, inputs_placeholder, embeddings):
    FACE_DB_PATH = config.FACE_DB_PATH
    face_db = []
    for root, dirs, files in os.walk(FACE_DB_PATH):
        for file in files:
            input_image = cv2.imdecode(np.fromfile(os.path.join(root, file), dtype=np.uint8), 1)
            try:
                input_image = input_image - 127.5
                input_image = input_image * 0.0078125
                name = file.split(".")[0]

                input_image = np.expand_dims(input_image, axis=0)

                feed_dict = {inputs_placeholder: input_image}
                emb_array = sess.run(embeddings, feed_dict=feed_dict)

                embedding = sklearn.preprocessing.normalize(emb_array).flatten()
                face_db.append({
                    "name": name,
                    "feature": embedding
                })
                print('loaded face: %s' % file)
            except Exception as e:
                print(e)
                print("delete error image:%s" % file)
                os.remove(os.path.join(root, file))
                continue
    return face_db


# 检测并裁剪人脸
def add_faces(mtcnn_detector):
    face_db_path = config.FACE_DB_PATH
    faces_name = os.listdir(face_db_path)
    temp_face_path = config.TEMP_FACE_PATH
    for root, dirs, files in os.walk(temp_face_path):
        for file in files:
            if file not in faces_name:
                input_image = cv2.imdecode(np.fromfile(os.path.join(root, file), dtype=np.uint8), 1)
                faces, landmarks = mtcnn_detector.detect(input_image)
                bbox = faces[0, :4]
                points = landmarks[0, :].reshape((5, 2))
                nimg = face_preprocess.preprocess(input_image, bbox, points, image_size='112,112')
                cv2.imwrite(os.path.join(face_db_path, os.path.basename(file)), nimg)


# 加载人脸识别模型
def load_mobilefacenet():
    MODEL_PATH = config.MOBILEFACENET_MODEL_PATH
    print('Model filename: %s' % MODEL_PATH)
    with tf.gfile.FastGFile(MODEL_PATH, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    # 获取人脸输入层
    inputs_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    # 获取人脸特征层输出
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    sess = tf.Session()
    return sess, inputs_placeholder, embeddings


# list 转成json格式数据
def list_to_json(lst):
    keys = [str(x) for x in np.arange(len(lst))]
    list_json = dict(zip(keys, lst))
    return list_json
