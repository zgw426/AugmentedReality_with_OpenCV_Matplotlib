"""
ControlNumber:03-06-04

概要
カメラ映像内のARマーカーにMatplotlibの3Dグラフをマッピングします。
透過処理により3Dグラフの背景を透明にします。

実行コマンド
python 04_matplot3d_transparency.py

qキーで終了します。
"""

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import datetime

targetVideo = 0 # カメラデバイス


def putSprite_npwhere(back, front4, pos):
    """透過処理"""
    x, y = pos
    fh, fw = front4.shape[:2]
    bh, bw = back.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x+fw, bw), min(y+fh, bh)
    if not ((-fw < x < bw) and (-fh < y < bh)) :
        return back
    front3 = front4[:, :, :3]
    front_roi = front3[y1-y:y2-y, x1-x:x2-x]
    roi = back[y1:y2, x1:x2]
    tmp = np.where((front_roi==(255,255,255))|(front_roi==(0,0,0)), roi, front_roi)  # 255x3:黒, 0x3:白
    back[y1:y2, x1:x2] = tmp

    return back


def graph_3d(roll_elev, pitch_azim):
    #---------------------------------------------#
    """グラフの計算式に関するもの"""
    dt_now = datetime.datetime.now()

    base_val = float(datetime.datetime.now().strftime('%f')[:-3])
    base_val =  round(base_val / 100, 2)

    base_val_x = dt_now.microsecond * 0.2
    base_val_y = dt_now.microsecond * 0.3

    # 円周率
    pi = np.pi

    # 分割数
    n = 32
    # 範囲-4pi～4piをn分割して(x,y)データを作成
    x = np.linspace(-4*pi+base_val_x, 4*pi+base_val_x, n)
    y = np.linspace(-4*pi+base_val_y, 4*pi+base_val_y, n)

    # 格子点の作成
    X, Y = np.meshgrid(x, y)
    # 高度の計算式
    Z = np.cos(np.sqrt(X**2 + Y**2))
    """グラフの計算式に関するもの"""

    plt.cla()  # 現在の軸をクリア
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=roll_elev, azim=pitch_azim)
    ax.set_axis_off()  # # 目盛り非表示
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)  # 余白なくす

    #ax.plot_surface(X,Y,z, cmap='terrain')  # 描画したいグラフ

    # 曲面を描画(カラーマップは'plasma_r')
    ax.plot_surface(X, Y, Z, cmap = "plasma_r")

    fig.canvas.draw()
    img_array = np.array(fig.canvas.renderer.buffer_rgba())
    img_return = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

    # メモリ解放
    plt.clf()
    plt.close()

    return img_return


def arReader(img, roll, pitch):
    im_src = graph_3d(roll, pitch)
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
            temp = cv2.warpPerspective(im_src.copy(), h, (img.shape[1], img.shape[0]))  # 射影変換
            img = putSprite_npwhere(img, temp, (0,0))
    return img


cap = cv2.VideoCapture(targetVideo, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 60)           # カメラFPSを60FPSに設定
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # カメラ画像の横幅を1280に設定
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # カメラ画像の縦幅を720に設定

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50) # ARマーカー
parameters = aruco.DetectorParameters_create()

cap_flg = 0
capW = 0
capH = 0
camera_matrix = 0

while cap.isOpened():
    # Capture frame-by-frame
    ret, img = cap.read()
    if img is None :
        break

    if cap_flg == 0:
        cap_flg = 1
        # 幅
        capW = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH) )
        # 高さ
        capH = int( cap.get(cv2.CAP_PROP_FRAME_HEIGHT) )

        center = (capW, capH)
        focal_length = center[0] / np.tan(60/2 * np.pi / 180)

        camera_matrix = np.array(
                    [[focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]], dtype = "double"
                )

    scale_percent = 50 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Check if frame is not empty
    if not ret:
        continue

    # Set AR Marker
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    
    parameters = aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion

    if len(corners) > 0:
        # マーカーごとに処理
        for i, corner in enumerate(corners):
            rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corner, 0.05, camera_matrix, dist_coeffs)
            # 不要なaxisを除去
            tvec = np.squeeze(tvecs)          # [[[ 0.00127975 -0.05548491  0.0717088 ]]] -> [ 0.00127975 -0.05548491  0.0717088 ]
            rvec = np.squeeze(rvecs)          # [-3.11190637e+00 -4.12759852e-01  8.28341191e-04]
            rvec_matrix = cv2.Rodrigues(rvec) # 回転ベクトルからrodoriguesへ変換
            rvec_matrix = rvec_matrix[0]      # rodoriguesから抜き出し

            # 回転ベクトルからrodoriguesへ変換
            rvec_matrix = cv2.Rodrigues(rvec) # 回転ベクトルからrodoriguesへ変換
            rvec_matrix = rvec_matrix[0]      # rodoriguesから抜き出し
            # 並進ベクトルの転置
            transpose_tvec = tvec[np.newaxis, :].T
            # 合成
            proj_matrix = np.hstack((rvec_matrix, transpose_tvec))
            # オイラー角への変換
            euler_angle = cv2.decomposeProjectionMatrix(proj_matrix)[6] # [deg]
            roll = np.squeeze(euler_angle[0])
            pitch = np.squeeze(euler_angle[1]) * -1
            yaw = np.squeeze(euler_angle[2])
            img = arReader(img, roll, pitch)

    # Display the resulting frame
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
