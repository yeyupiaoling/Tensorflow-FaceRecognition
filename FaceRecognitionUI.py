import cv2
import numpy as np
import sklearn
import config
from utils import face_preprocess
from PIL import Image, ImageTk, ImageFont, ImageDraw
from utils.utils import feature_compare, load_mtcnn, load_faces, load_mobilefacenet, add_faces
import tkinter as tk

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

# 全局变量用于保存视频捕获对象和当前显示的图像
cap_register = None
cap_recognition = None
current_image_register = None
current_image_recognition = None
# 用于标记是否正在注册过程中
is_registering = False
# 用于保存注册时输入的用户名
register_name = ""


# 注册人脸
def face_register():
    global cap_register, current_image_register, is_registering
    print("点击'注册人脸'按钮后，输入名字，再点击'拍照注册' 进行拍照！")
    cap_register = cv2.VideoCapture(0)
    is_registering = True
    update_register_video()  # 开始更新注册视频显示的循环


def update_register_video():
    global cap_register, current_image_register, is_registering
    if cap_register is None or not cap_register.isOpened() or not is_registering:
        return
    ret, frame = cap_register.read()
    if ret:
        # 直接将彩色的原始帧转换为可用于tkinter显示的格式
        current_image_register = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        register_video_label.configure(image=current_image_register)
    root.after(10, update_register_video)  # 每隔10毫秒更新一次图像显示


# 处理注册拍照及保存操作
def take_register_photo():
    global is_registering, register_name
    is_registering = False
    ret, frame = cap_register.read()
    if ret:
        faces, landmarks = mtcnn_detector.detect(frame)
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
                nimg = face_preprocess.preprocess(frame, bbox, points, image_size='112,112')
                user_name = register_name.get()  # 获取输入框中的注册名
                cv2.imencode('.png', nimg)[1].tofile('face_db/%s.png' % user_name)
                print("注册成功！")
            else:
                print('注册图片有错，图片中有且只有一个人脸')
        else:
            print('注册图片有错，图片中有且只有一个人脸')
    cap_register.release()


# 人脸识别
def face_recognition():
    global cap_recognition, current_image_recognition
    cap_recognition = cv2.VideoCapture(0)
    update_recognition_video()  # 开始更新识别视频显示的循环


def update_recognition_video():
    global cap_recognition, current_image_recognition
    if cap_recognition is None or not cap_recognition.isOpened():
        return
    ret, frame = cap_recognition.read()
    if ret:
        faces, landmarks = mtcnn_detector.detect(frame)
        if faces.shape[0] is not 0:
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
                    nimg = face_preprocess.preprocess(frame, bbox, points, image_size='112,112')
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
                # 写上人脸信息，这里确保处理后的图像为彩色格式用于显示
                x1, y1, x2, y2 = faces[k][0], faces[k][1], faces[k][2], faces[k][3]
                x1 = max(int(x1), 0)
                y1 = max(int(y1), 0)
                x2 = min(int(x2), frame.shape[1])
                y2 = min(int(y2), frame.shape[0])
                prob = '%.2f' % probs[k]
                label = "{}, {}".format(info_name[k], prob)
                cv2img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pilimg = Image.fromarray(cv2img)
                draw = ImageDraw.Draw(pilimg)
                font = ImageFont.truetype('font/simfang.ttf', 18, encoding="utf-8")
                draw.text((x1, y1 - 18), label, (255, 0, 0), font=font)
                frame = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # 将处理后的彩色cv2图像转换为tkinter可用的图像格式
            current_image_recognition = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            recognition_video_label.configure(image=current_image_recognition)
    root.after(10, update_recognition_video)  # 每隔10毫秒更新一次图像显示


def stop_recognition():
    global cap_recognition
    if cap_recognition is not None:
        cap_recognition.release()
        cap_recognition = None


def on_register_click():
    face_register()


def on_recognition_click():
    face_recognition()


if __name__ == '__main__':
    root = tk.Tk()
    root.title("人脸识别系统")
    root.geometry("1600x600")

    register_video_label = tk.Label(root)
    register_video_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    recognition_video_label = tk.Label(root)
    recognition_video_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # 新增拍照注册按钮
    take_photo_button = tk.Button(root, text="拍照注册", command=take_register_photo)
    take_photo_button.pack(side=tk.BOTTOM, padx=20, pady=20)

    # 新增输入框，用于输入注册名
    register_name = tk.StringVar()
    name_entry = tk.Entry(root, textvariable=register_name)
    name_entry.pack(side=tk.BOTTOM, padx=20, pady=20)

    register_button = tk.Button(root, text="注册人脸", command=on_register_click)
    register_button.pack(side=tk.BOTTOM, padx=20, pady=20)

    
    # 新增停止识别按钮
    stop_button = tk.Button(root, text="停止识别", command=stop_recognition)
    stop_button.pack(side=tk.BOTTOM, padx=20, pady=20)

    recognition_button = tk.Button(root, text="识别人脸", command=on_recognition_click)
    recognition_button.pack(side=tk.BOTTOM, padx=20, pady=20)

    root.mainloop()