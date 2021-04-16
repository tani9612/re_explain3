from tkinter import Image

import cv2
import numpy as np
import copy
from matplotlib import pyplot as plt
import math

np.seterr(divide='ignore')

bug = np.array(0)
divided = 1/bug
# print(divided)

N = 100  # 仮
size = 150
# p = 0.9

path_w = '/Users/tanioka/PycharmProjects/re_explain3/a.txt'
with open(path_w, mode='w') as f:

    # (1)
    img2 = cv2.imread("image4-2.jpg", 0)
    img = img2/255
    X2 = cv2.imread("image4-2.jpg", 0)
    X = X2/255

    row, col = img.shape
    # ノイズ黒
    pts_x = np.random.randint(0, col-1, 1000)
    pts_y = np.random.randint(0, row-1, 1000)
    img[pts_y, pts_x] = 0
    # ばってん部分
    img = cv2.line(img, (0, 0), (size-1, size-1), (0, 0, 0), 5)
    img = cv2.line(img, (0, size-1), (size-1, 0), (0, 0, 0), 5)

    img_array = np.array(img)
    X_array = np.array(X)
    cv2.imwrite("image_noise.jpg", img*255)
    # ノイズ部分取得OK
    list_PSNR = []
    list_k = []
    v2 = np.random.uniform(0, size, (1, size))
    for p in np.arange(0.1, 2.1, 0.1):
        p_str = str('{:.2f}'.format(p))
        f.write('\np=')
        f.write(p_str)
        f.flush()
        r = img_array
        r_new = np.zeros((size, size))
        M = np.zeros((size, size))
# 0.1~2のループ入れる
        for k in range(40):  # 一番外側　ランク
            v = copy.deepcopy(v2)
            u = np.zeros((size, 1))
            a = np.zeros((1, size))
            b = np.zeros((1, size))
            c = np.zeros((1, size))
            d = np.zeros((1, size))
            w = np.zeros((1, size))


            E = 50
            print('k=', k)
            for q in range(30):  # 交互反復のループ
                print('q=', q)
                for i in range(size):  # uの要素数
                    a = []
                    b = []
                    count = 0
                    for x in range(size):
                        if img_array[i, x] != 0:
                            a.append(r[i, x])
                            b.append(v[0, x])
                            count += 1
                    a_np = np.array(a).reshape(1, count)
                    b_np = np.array(b).reshape(1, count)
                    u[i, 0] = (b_np@a_np.T)/(b_np@b_np.T)  # uのIRLSの0回目
                    for t in range(30):  # IRLSのループ
                        tau = a_np - u[i, 0] * b_np
                        tau_max = np.maximum(np.abs(tau), 1e-4)
                        w = 1 / (np.abs(tau_max) ** ((2 - p) / 2))  # wの計算
                        x_0 = np.sum(2*w*w*a_np*b_np)
                        x_1 = np.sum(2*w*w*b_np*b_np)
                        u[i, 0] = x_0/x_1

                for i in range(size):  # vの要素数
                    c = []
                    d = []
                    count2 = 0
                    for x in range(size):
                        if img_array[i, x] != 0:
                            c.append(r[x, i])
                            d.append(u[x, 0])
                            count2 += 1
                    c_np = np.array(c).reshape(1, count2)
                    d_np = np.array(d).reshape(1, count2)
                    v[0, i] = (d_np @ c_np.T) / (d_np @ c_np.T)  # uのIRLSの0回目
                    for t in range(30):  # IRLSのループ
                        tau = c_np - d_np * v[0, i]
                        tau_max = np.maximum(np.abs(tau), 1e-4)
                        w = 1 / (np.abs(tau_max) ** ((2 - p) / 2))  # wの計算
                        x_0 = np.sum(2*w*w*c_np*d_np)
                        x_1 = np.sum(2*w*w*d_np*d_np)
                        v[0, i] = x_0 / x_1
                a_flat = (r - (u @ v)).flatten()
                b_flat = r.flatten()
                E_new = (np.linalg.norm(a_flat, ord=2)) ** 2 / (np.linalg.norm(b_flat, ord=2)) ** 2
                # qの収束条件
                sigma = np.abs(E - E_new)
                E = copy.copy(E_new)
            r_new = r - (u @ v)
            for y in range(size):
                for x in range(size):
                    if img_array[y, x] == 0:
                        r_new[y, x] = 0
            M += u @ v
            # kの収束条件
            c_flat = r.flatten()
            d_flat = img.flatten()
            e_flat = r_new.flatten()
            eta = np.abs((np.linalg.norm(c_flat-e_flat, ord=2))**2/(np.linalg.norm(d_flat, ord=2))**2)
            print('eta=', eta)
            eta_str = str(eta)
            f.write('\neta=')
            f.write(eta_str)
            f.flush()
            if eta < 5 * 10**(-4):
                print('k=', k)
                break
            r = copy.deepcopy(r_new)
        #Mの要素を0~255に絞る（力技）
        M[M < 0] = 0
        M[M > 1] = 1
        print('max=', np.amax(M))
        print('min=', np.amin(M))

        mse = ((np.linalg.norm(X_array-M, ord=2))**2)/(size*size)
        PSNR = 10*np.log10((255**2)/mse)
        list_PSNR.append(PSNR)
        list_k.append(k)
        print('PSNR=', PSNR)
        M2 = 255*M

        cv2.imwrite(p_str + ".jpg", M2)
        k_str = str(k)
        PSNR_str = str(PSNR)
        f.write('\nk=')
        f.write(k_str)
        f.write('\nPSNR=')
        f.write(PSNR_str)
        f.flush()

    print(list_PSNR)
    PSNR_x = np.arange(0.1, 2.1, 0.1)
    PSNR_y = list_PSNR
    plt.plot(PSNR_x, PSNR_y)
    plt.savefig("PSNR.png")
    plt.plot(PSNR_x, list_k)
    plt.savefig("rank.png")

