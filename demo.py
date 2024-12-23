
import cv2
import numpy as np
import sklearn
import config
from utils import face_preprocess
from PIL import ImageFont, ImageDraw, Image
from utils.utils import feature_compare, load_mtcnn, load_faces, load_mobilefacenet, add_faces
import streamlit as st

# 人脸识别阈值
VERIFICATION_THRESHOLD = config.VERIFICATION_THRESHOLD

# 检测人脸检测模型
mtcnn_detector = load_mtcnn()
# 加载人脸识别模型
face_sess, inputs_placeholder, embeddings = load_mobilefacenet()
# 添加人脸
add_faces(mtcnn_detector)
# 加载已经注册的人脸
faces_db = load_faces(face_sess, inputs_placeholder, embeddings)


# 注册人脸
def face_register():
    st.write("点击 y 确认拍照！")
    # 利用 Streamlit 的相机输入
    img_file_buffer = st.camera_input("拍照")
    if img_file_buffer is not None:
        # 将图像转换为 OpenCV 格式
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        faces, landmarks = mtcnn_detector.detect(cv2_img)
        if faces.shape[0]!= 0:
            faces_sum = 0
            bbox = []
            points = []
            for i, face in enumerate(faces):
                if round(faces[i, 4], 6) > 0.95:
                    bbox = faces[i, 0:4]
                    points = landmarks[i, :].reshape((5, 2))
                    faces_sum += 1
            if faces_sum == 1:
                nimg = face_preprocess.preprocess(cv2_img, bbox, points, image_size='112,112')
                user_name = st.text_input("请输入注册名：")
                if user_name:
                    cv2.imencode('.png', nimg)[1].tofile('face_db/%s.png' % user_name)
                    st.write("注册成功！")
            else:
                st.write('注册图片有错，图片中有且只有一个人脸')
        else:
            st.write('注册图片有错，图片中有且只有一个人脸')


# 人脸识别
def face_recognition():
    img_file_buffer = st.camera_input("拍照")
    if img_file_buffer is not None:
        # 将图像转换为 OpenCV 格式
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        faces, landmarks = mtcnn_detector.detect(cv2_img)
        if faces.shape[0]!= 0:
            faces_sum = 0
            for i, face in enumerate(faces):
                if round(faces[i, 4], 6) > 0.95:
                    faces_sum += 1
            if faces_sum == 0:
                return
            # 人脸信息
            info_location = np.zeros(faces_sum)
            info_location[0] = 1
            info_name = []
            probs = []
            # 提取图像中的人脸
            input_images = np.zeros((faces.shape[0], 112, 112, 3))
            for i, face in enumerate(faces):
                if round(faces[i, 4], 6) > 0.95:
                    bbox = faces[i, 0:4]
                    points = landmarks[i, :].reshape((5, 2))
                    nimg = face_preprocess.preprocess(cv2_img, bbox, points, image_size='112,112')
                    nimg = nimg - 127.5
                    nimg = nimg * 0.0078125
                    input_images[i, :] = nimg
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
            for k in range(faces_sum):
                # 写上人脸信息
                x1, y1, x2, y2 = faces[k][0], faces[k][1], faces[k][2], faces[k][3]
                x1 = max(int(x1), 0)
                y1 = max(int(y1), 0)
                x2 = min(int(x2), cv2_img.shape[1])
                y2 = min(int(y2), cv2_img.shape[0])
                prob = '%.2f' % probs[k]
                label = "{}, {}".format(info_name[k], prob)
                cv2img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                pilimg = Image.fromarray(cv2img)
                draw = ImageDraw.Draw(pilimg)
                font = ImageFont.truetype('font/simfang.ttf', 18, encoding="utf-8")
                draw.text((x1, y1 - 18), label, (255, 0, 0), font=font)
                cv2_img = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
                cv2.rectangle(cv2_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        st.image(cv2_img, channels="BGR")


def main():
    st.title("人脸识别系统")
    option = st.selectbox(
        '请选择功能',
        ('注册人脸', '识别人脸'))
    if option == '注册人脸':
        face_register()
    elif option == '识别人脸':
        face_recognition()


if __name__ == '__main__':
    main()
