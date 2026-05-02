import sys
import numpy as np

from PIL import Image,ImageDraw

import tensorflow as tf

from tensorflow.keras import layers , models

#数据增强
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau

import os

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QMessageBox,
                             QHBoxLayout)

from PyQt5.QtGui import QPainter,QPen,QColor,QImage

from PyQt5.QtCore import Qt,QPoint

#全局变量
MODEL_PATH = 'digit_model.h5'

#神经网络模型
def create_model():
    model = models.Sequential(
        [
            #第一层卷积：32个 3*3卷积核
         layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
         layers.Conv2D(32, (3, 3), activation='relu'),
            #池化层
         layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            #第二层卷积 64个卷积核
         layers.Conv2D(64, (3, 3), activation='relu'),
         layers.Conv2D(64, (3, 3), activation='relu'),
         layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            #第三层卷积层
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.Dropout(0.25),
            #展平层
            layers.Flatten(),
            #全连接层 128个神经元
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),#丢弃率更高，防止过拟合
            #输出层
            layers.Dense(10, activation='softmax')
         ]
    )

#使用Adam优化器，学习率设为0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#编译模型
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

#训练模型
def train_model():
    print('正在加载MNIST数据集..')
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    # 数据预处理
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
    #数据增强
    datagen = ImageDataGenerator(
        rotation_range=10,#随机旋转±10度
        zoom_range=0.1,#随机缩放±0.1
        width_shift_range=0.1,#水平平移±10度
        height_shift_range=0.1,#垂直平移±10度
    )
    datagen.fit(train_images)

    #创建模型
    model = create_model()

    #学习率衰减回调
    learning_rate_reduction = ReduceLROnPlateau(
        monitor='val_accuracy',
        patience=3,
        verbose=1,
        factor=0.5,
        min_lr=0.00001,
    )
    #开始训练
    #用20个epoch应该够了
    #先改为1个，不然又报错了
    print('开始训练模型')
    model.fit(
        datagen.flow(
            train_images,
            train_labels,
            batch_size=64,
        ),
        epochs=20,
        validation_data=(test_images, test_labels),
        callbacks=[learning_rate_reduction],
    )

    #用测试训练集来评估新能
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'测试准确率:{test_acc:.4f}')

    #保存模型
    model.save(MODEL_PATH)#保留完整模型而不是只保存权重
    print("模型保存成功")
    return model
#加载或训练模型
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print('模型加载成功')
            return model
        except Exception as e:
            print('模型文件不存在，开始训练')
    return train_model()


#自定义画布类
#可以创建一个画布来让鼠标手写数字
class DrawingCanvas(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        #设置画布大小为280*280
        self.setFixedSize(280,280)
        self.image = Image.new('L', (280, 280),255)#"L"代表灰度，255表示白色背景
        self.draw = ImageDraw.Draw(self.image)

        #记录上一次的位置，来绘制连续的线
        self.last_point = QPoint()

        #设置个样式，灰色边框白色背景
        self.setStyleSheet("background-color:white;border:2px solid gray;")

        #获取当前画布上的图像
    def get_image(self):
        return self.image

        #清空画布
    def clear(self):
        self.image = Image.new('L', (280, 280),255)
        self.draw = ImageDraw.Draw(self.image)
        self.update()#刷新显示

        #绘制事件
    def paintEvent(self, event):
        painter = QPainter(self)
        qimage = QImage(self.image.tobytes(),280,280,QImage.Format.Format_Grayscale8)
        painter.drawImage(0,0,qimage)

        #鼠标按下事件
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_point = event.pos()

        #鼠标移动事件
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.draw.line([self.last_point.x(), self.last_point.y(),event.x(),event.y()],fill=0,width=15)
            self.last_point = event.pos()
            self.update()
#主窗口类
class MainWindow(QMainWindow):
    def __init__(self,model):#少接受了model
        super().__init__()

        self.setWindowTitle('手写数字识别')#窗口标题

        self.model = model #保存模型引用，后面预测用

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)
        layout.setAlignment(Qt.AlignCenter)

        self.canvas = DrawingCanvas()
        layout.addWidget(self.canvas)

        button_layout = QVBoxLayout()

        self.predict_bth = QPushButton('识别数字')
        self.predict_bth.setStyleSheet("font-size:14px;padding: 8px 20px;")
        self.predict_bth.clicked.connect(self.predict)
        button_layout.addWidget(self.predict_bth)

        self.clear_btn = QPushButton('清除画布')
        self.clear_btn.clicked.connect(self.canvas.clear)
        self.clear_btn.setStyleSheet("font-size:14px;padding: 8px 20px;")
        button_layout.addWidget(self.clear_btn)

        layout.addLayout(button_layout)

        self.result_label = QLabel('识别结果：')
        self.result_label.setStyleSheet("font-size:20px;font-weight:bold;")
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)

        self.confidence_label = QLabel('置信度：')
        self.confidence_label.setStyleSheet("font-size:14px;")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.confidence_label)

        self.setStyleSheet("QMainWindow{background-color:#f0f0f0;}")

    #预测方法
    def predict(self):
        img = self.canvas.get_image()
        img = img.resize((28, 28)),
        img_array = np.array(img)
        img_array = 255 - img_array#颜色反转
        img_array = img_array.astype('float32')/255 #归一化
        img_array = img_array.reshape(1,28,28,1)
        prediction = self.model.predict(img_array,verbose=0)
        digit = np.argmax(prediction)
        confidence = prediction[0][digit] *100

        self.result_label.setText(f'识别结果:{digit}')
        self.confidence_label.setText(f'置信度:{confidence:.2f}%')

#程序入口
if __name__ == '__main__':
    print('='*50)
    print('手写数字识别工具')
    print('='*50)
#退出后才出现h5文件
    #加载或训练模型
    model = load_or_train_model()

    #创建QT应用对象
    app = QApplication(sys.argv)

    #创建主窗口
    window = MainWindow(model)

    #显示
    window.show()

    #循环
    sys.exit(app.exec_())
