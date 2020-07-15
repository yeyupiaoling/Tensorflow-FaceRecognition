import config
import time
import cv2
import numpy as np
import sklearn
from flask import request, Flask, render_template
from flask_cors import CORS
from utils import face_preprocess
from utils.utils import feature_compare, load_mtcnn, load_faces, load_mobilefacenet, list_to_json

app = Flask(__name__)
# 允许跨越访问
CORS(app)

# 加载人脸检测模型
VERIFICATION_THRESHOLD = config.VERIFICATION_THRESHOLD
# 检测人脸检测模型
mtcnn_detector = load_mtcnn()
# 加载人脸识别模型
face_sess, inputs_placeholder, embeddings = load_mobilefacenet()
# 加载已经注册的人脸
faces_db = load_faces(face_sess, inputs_placeholder, embeddings)


# 人脸识别
def recognition_face(frame):
    try:
        faces, landmarks = mtcnn_detector.detect(frame)
        if faces.shape[0] is not 0:
            faces_sum = 0
            for i, face in enumerate(faces):
                if round(faces[i, 4], 6) > 0.95:
                    faces_sum += 1
            if faces_sum == 0:
                return
            # 人脸信息
            info_bbox = np.zeros((faces_sum, 4))
            info_landmarks = np.zeros((faces_sum, 10))
            info_name = []
            probs = []
            img_i = 0
            # 提取图像中的人脸
            input_images = np.zeros((faces.shape[0], 112, 112, 3))
            for i, face in enumerate(faces):
                if round(faces[i, 4], 6) > 0.95:
                    bbox = faces[i, 0:4]
                    points = landmarks[i, :].reshape((5, 2))
                    nimg = face_preprocess.preprocess(frame, bbox, points, image_size='112,112')
                    # 图像预处理
                    nimg = nimg - 127.5
                    nimg = nimg * 0.0078125
                    input_images[i, :] = nimg
                    # 关键点和人脸框
                    info_bbox[img_i] = bbox
                    info_landmarks[img_i] = landmarks[i, :]
                    img_i += 1

            # 进行人脸识别
            feed_dict = {inputs_placeholder: input_images}
            emb_arrays = face_sess.run(embeddings, feed_dict=feed_dict)
            emb_arrays = sklearn.preprocessing.normalize(emb_arrays)
            for i, embedding in enumerate(emb_arrays):
                embedding = embedding.flatten()
                temp_dict = {}
                # 比较已经存在的人脸数据库
                for com_face in faces_db:
                    ret, sim = feature_compare(embedding, com_face["feature"], 0.70)
                    temp_dict[com_face["name"]] = sim
                dict = sorted(temp_dict.items(), key=lambda d: d[1], reverse=True)
                if dict[0][1] > VERIFICATION_THRESHOLD:
                    name = dict[0][0]
                    probs.append(dict[0][1])
                    info_name.append(name)
                else:
                    probs.append(dict[0][1])
                    info_name.append("unknown")
            # 如果识别就跳出检测
            return info_name, probs, info_bbox, info_landmarks
        else:
            raise Exception("infer result is None!")
    except Exception as e:
        print(e)
        return None, None, None, None, None, None


# 识别人脸接口
@app.route("/recognition", methods=['POST'])
def recognition():
    start_time1 = time.time()
    upload_file = request.files['image']
    is_chrome_camera = request.values.get("is_chrome_camera")
    if upload_file:
        try:
            img = cv2.imdecode(np.frombuffer(upload_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            # 兼容浏览器摄像头拍照识别
            if is_chrome_camera == "True":
                cv2.imwrite('test.png', img)
                img = cv2.imdecode(np.fromfile('test.png', dtype=np.uint8), 1)
        except:
            return str({"error": 2, "msg": "this file is not image"})
        try:
            info_name, probs, info_bbox, info_landmarks = recognition_face(img)
            if info_name is None:
                return str({"error": 3, "msg": "image not have face"})
        except:
            return str({"error": 3, "msg": "image not have face"})
        # 封装识别结果
        data_faces = []
        for i in range(len(info_name)):
            data_faces.append(
                {"name": info_name[i], "probability": probs[i],
                 "bbox": list_to_json(np.around(info_bbox[i], decimals=2).tolist()),
                 "landmarks": list_to_json(np.around(info_landmarks[i], decimals=2).tolist())})
        data = str({"code": 0, "msg": "success", "data": data_faces}).replace("'", '"')
        print('duration:[%.0fms]' % ((time.time() - start_time1) * 1000), data)
        return data
    else:
        return str({"error": 1, "msg": "file is None"})


# 注册人脸接口
@app.route("/register", methods=['POST'])
def register():
    global faces_db
    upload_file = request.files['image']
    user_name = request.values.get("name")
    if upload_file:
        try:
            image = cv2.imdecode(np.frombuffer(upload_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            faces, landmarks = mtcnn_detector.detect(image)
            if faces.shape[0] is not 0:
                faces_sum = 0
                bbox = []
                points = []
                for i, face in enumerate(faces):
                    if round(faces[i, 4], 6) > 0.95:
                        bbox = faces[i, 0:4]
                        points = landmarks[i, :].reshape((5, 2))
                        faces_sum += 1
                if faces_sum == 1:
                    nimg = face_preprocess.preprocess(image, bbox, points, image_size='112,112')
                    cv2.imencode('.png', nimg)[1].tofile('face_db/%s.png' % user_name)
                    # 更新人脸库
                    faces_db = load_faces(face_sess, inputs_placeholder, embeddings)
                    return str({"code": 0, "msg": "success"})
            return str({"code": 3, "msg": "image not or much face"})
        except:
            return str({"code": 2, "msg": "this file is not image or not face"})
    else:
        return str({"code": 1, "msg": "file is None"})


@app.route('/')
def home():
    return render_template("index.html")


if __name__ == '__main__':
    app.run(host=config.HOST, port=config.POST)
