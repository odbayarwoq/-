# - Description

Энэ хүү төслийн хүрээнд хяналтийн камерийн бичлэгээс өгөгдөл цуглуулан. Цурглуулсан өгөгдлөө YuNet царай илрүүлэх моделийг сонгон ашигласан. Ингэснээр өөрийн цуглуулсан өгөгдлийн санг ашиглан машин сургалтын аргаар боловсруулан царай таних загваруудыг сургаж авсан. Өөрийн сургасан моделийг ашиглан амжилттай царай танилтыг хийсэн болно.

Царай илрүүлхэд ашиглах тэмдэглэл:

- Загварын эх сурвалж: [here](https://github.com/ShiqiYu/libfacedetection.train/blob/a61a428929148171b488f024b5d6774f93cdbc13/tasks/task1/onnx/yunet.onnx).
- Энэ загвар нь сургалтын схемийн улмаас **ойролцоогоор 10x10-аас 300x300** хүртэлх пикселийн нүүрийг илрүүлэх боломжтой.
- Энэ загварыг сургах талаар дэлгэрэнгүй мэдээллийг https://github.com/ShiqiYu/libfacedetection.train хаягаар авна уу.
- Энэ ONNX загвар нь тогтмол оролтын хэлбэртэй боловч OpenCV DNN нь оролтын зургийн яг хэлбэрийг гаргадаг. Дэлгэрэнгүй мэдээллийг https://github.com/opencv/opencv_zoo/issues/44-с үзнэ үү.

# - Installation

```shell
pip install opencv-python
pip install skorch
pip install efficientnet_pytorch
pip install torch
pip install torchvision
```
