"""
ControlNumber:03-05-05-01

概要
入力画像(sample.png)のARマーカーにMatplotlibの2Dグラフをマッピングし、その結果を出力画像(out-sample.png)として出力します。

実行コマンド
python 01_matplot2d_not-camera.py

inputfile : sample.png
outputfile : out-sample.png
"""
import cv2
import numpy as np
from PIL import Image
import math
from matplotlib import pyplot as plt
import random

targetImg  = "sample.png"
outputImg  = "out-" + targetImg[0:-4]+".png"


def graph_2d():
    random_no = random.randrange(8)
    background_color = colorlist[random.randrange(len(colorlist))]
    x = np.linspace(0, 2*pi, 100)  #0から2πまでの範囲を100分割したnumpy配列
    y = np.sin(x + random_no)
    fig, ax = plt.subplots()
    ax = plt.axes()
    ax.set_facecolor(background_color)
    ax.plot(x, y, 'r,-.')
    fig.canvas.draw()
    img_array = np.array(fig.canvas.renderer.buffer_rgba())
    img_return = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    return img_return


def arReader():
    img = cv2.imread( targetImg ) # ARマーカーを含む画像
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_src = graph_2d()

    # ARマーカを検出
    ## type(ids)= <class 'numpy.ndarray'> ※ARマーカ―検出
    ## type(ids)= <class 'NoneType'>      ※ARマーカ―未検出
    ## corners: 検出した各ARマーカーの4隅の座標
    corners, ids, _ = aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    
    if np.all(ids != None):
        # 検出したARマーカーの数ループする
        for c in corners :
            x1 = (c[0][0][0], c[0][0][1]) 
            x2 = (c[0][1][0], c[0][1][1]) 
            x3 = (c[0][2][0], c[0][2][1]) 
            x4 = (c[0][3][0], c[0][3][1])   
 
            size = im_src.shape
            pts_dst = np.array([x1, x2, x3, x4])
            pts_src = np.array(
                           [
                            [0,0],
                            [size[1] - 1, 0],
                            [size[1] - 1, size[0] -1],
                            [0, size[0] - 1 ]
                           ],dtype=float
                        )
            h, status = cv2.findHomography(pts_src, pts_dst)
            temp = cv2.warpPerspective(im_src.copy(), h, (img.shape[1], img.shape[0])) 
            cv2.fillConvexPoly(img, pts_dst.astype(int), 0, 16)
            img = cv2.add(img , temp)
            aruco.drawDetectedMarkers(img, corners, ids, (255,0,0)) #検出したマーカに,マッピング用の画像を描画する
    img = Image.fromarray(img)
    img.save( outputImg )
    print("DONE : \n\tinputfile is {0}\n\toutputfile is {1}".format(targetImg, outputImg))


aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

pi = math.pi  # mathモジュールのπを利用
colorlist = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#FFFFFF']

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50) # ARマーカー
parameters = aruco.DetectorParameters_create()

arReader()
