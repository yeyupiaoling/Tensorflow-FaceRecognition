# 人脸识别
 - `camera_demo.py` 调用摄像头进行人脸识别，并通过视频展示
 - `camera_main.py` 通过HTTP接口调用启动摄像头，并进行识别，把识别结果返回
 - `server_main.py` 人脸识别，通过HTTP调用，接收传过来的图片进行识别并返回结果
 
 
```
loaded face: 张伟.png
loaded face: 迪丽热巴.png
请选择功能，1为注册人脸，2为识别人脸：1
请输入图片路径：test.png
请输入注册名：夜雨飘零
注册成功！
```

```
loaded face: 张伟.png
loaded face: 迪丽热巴.png
请选择功能，1为注册人脸，2为识别人脸：1
点击y确认拍照！
请输入注册名：夜雨飘零

注册成功！
```

```
loaded face: 张伟.png
loaded face: 迪丽热巴.png
 * Serving Flask app "server_main" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://localhost:5000/ (Press CTRL+C to quit)
```