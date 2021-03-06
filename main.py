from tkinter import Image

import cv2
import numpy as np
import copy
import math
import skimage.metrics
from matplotlib import pyplot as plt
fig, ax = plt.subplots()

np.seterr(divide='ignore')

bug = np.array(0)
divided = 1/bug
# print(divided)

N = 100  # 仮
size = 150
# c = int(2)  # クロスバリデーションの繰り返し回数
# p = 0.9

path_w = '/Users/tanioka/PycharmProjects/re_explain3/a.txt'
with open(path_w, mode='w') as f:

    # (1)
    img2 = cv2.imread("image2-2.jpg", 0)
    img = img2/255
    X2 = cv2.imread("image2-2.jpg", 0)
    X = X2/255

    row, col = img.shape
    # ノイズ黒
    # 150*150=3557
    # img5(712*712)=8000(適当)
    pts_x = np.random.randint(0, col-1, 3557)
    pts_y = np.random.randint(0, row-1, 3557)
    img[pts_y, pts_x] = 0.1

    # ノイズ白（未）
    pts_x = np.random.randint(0, col-1, 3557)
    pts_y = np.random.randint(0, row-1, 3557)
    img[pts_y, pts_x] = 1

    """
    # 円部分
    img = cv2.circle(img, (75, 75), 50, (0, 0, 0), thickness=5)
    """

    """
    # テキスト
    img = cv2.putText(img, 'iiguni', (25, 75), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0), thickness=2)
    """

    # ばってん部分
    # 150*150=5
    # img5(712*712)=10(適当)
    img = cv2.line(img, (0, 0), (size-1, size-1), (0, 0, 0), 5)
    img = cv2.line(img, (0, size-1), (size-1, 0), (0, 0, 0), 5)

    """
    # 木隠してみる
    img = cv2.rectangle(img, (10, 40), (40, 130), (0, 0, 0), thickness=-1)
    """

    # Numpy配列の作成
    img_array = np.array(img)
    X_array = np.array(X)
    cv2.imwrite("image_noise.jpg", img*255)

    list_PSNR = []
    list_SSIM = []
    # 多分いらん下に書いてある
    # list_MAPE = []
    list_MAPE_sum = []
    list_SSIM_closs_sum = []
    list_PSNR_closs_sum = []
    list_k = []
    v2 = np.random.uniform(0, 1, (1, size))
    # pの値（0.1~2.0）
    """
    元画像わかってるやつ
    """
    for p in np.arange(0.1, 2.1, 0.1):
        # 小数点を調整
        p_str = str('{:.2f}'.format(p))
        f.write('\np=')
        f.write(p_str)
        f.flush()
        # r=差
        r = img_array
        r_new = np.zeros((size, size))
        # M=最終結果
        M = np.zeros((size, size))

        for k in range(40):  # 一番外側　ランク
            # 色々な初期化
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
                for i in range(size):  # uの要素数
                    a = []
                    b = []
                    count = 0
                    # ノイズ入ってない部分だけ計算
                    for x in range(size):
                        if img_array[i, x] != 0.1:
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
                # 多次元を1次元に
                a_flat = (r - (u @ v)).flatten()
                b_flat = r.flatten()
                E_new = (np.linalg.norm(a_flat, ord=2)) ** 2 / (np.linalg.norm(b_flat, ord=2)) ** 2
                # qの収束条件
                sigma = np.abs(E - E_new)
                E = copy.copy(E_new)
            uv = u @ v

            # 思ってたより結果よくならなかったからボツ
            # Mの要素を0~255に絞る（力技）
            # uv[uv < 0] = 0
            # uv[uv > 1] = 1

            r_new = r - uv
            for y in range(size):
                for x in range(size):
                    if img_array[y, x] == 0:
                        r_new[y, x] = 0
            M += uv
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
        # Mの分布図
        """
        plt.scatter(range(22500), M*255, s=3)
        plt.grid()
        plt.savefig(p_str + "M.png")
        # pltの初期化
        plt.clf()
        plt.cla()
        plt.close()
        # Mのup画像
        plt.scatter(range(22500), M * 255, s=3)
        plt.grid()
        plt.ylim(-300, 600)
        plt.savefig(p_str + "M_up.png")
        # pltの初期化
        plt.clf()
        plt.cla()
        plt.close()
        """

        # Mの要素を0~255に絞る（力技）
        M[M < 0] = 0
        M[M > 1] = 1
        # print('max=', np.amax(M))
        # print('min=', np.amin(M))

        M2 = 255*M
        cv2.imwrite(p_str + ".jpg", M2)

        """
        # 手動PSNR
        mse = ((np.linalg.norm(X_array-M, ord=2))**2)/(size*size)
        PSNR = 10*np.log10((255**2)/mse)
        """

        # 関数PSNR
        # めんどいけど画像にしてからもっかい呼び出す
        img_M = cv2.imread(p_str + ".jpg", 0)
        # PSNR出すよ
        PSNR = cv2.PSNR(img2, img_M)

        # SSIM出すよ
        SSIM = skimage.metrics.structural_similarity(img2, img_M)
        print('PSNR=', PSNR)
        print('SSIM=', SSIM)

        list_PSNR.append(PSNR)
        list_SSIM.append(SSIM)
        list_k.append(k)
        k_str = str(k)
        PSNR_str = str(PSNR)
        SSIM_str = str(SSIM)
        f.write('\nk=')
        f.write(k_str)
        f.write('\nPSNR=')
        f.write(PSNR_str)
        f.write('\nSSIM=')
        f.write(SSIM_str)
        f.flush()
    print(list_PSNR)
    PSNR_x = np.arange(0.1, 2.1, 0.1)
    PSNR_y = list_PSNR

    plt.title('PSNR')
    plt.grid(True)
    plt.xlabel('p')
    plt.ylabel('PSNR')
    plt.plot(PSNR_x, PSNR_y, marker="o")
    plt.ylim(10.0, 25.0)
    plt.xticks(np.arange(0.1, 2.1, 0.1))
    plt.savefig("PSNR.png")

    # pltの初期化
    plt.clf()
    plt.cla()
    plt.close()

    plt.title('SSIM')
    plt.grid(True)
    plt.xlabel('p')
    plt.ylabel('SSIM')
    plt.plot(PSNR_x, list_SSIM, marker="o")
    plt.ylim(0, 1.0)
    plt.xticks(np.arange(0.1, 2.1, 0.1))
    plt.savefig("SSIM.png")

    # pltの初期化
    plt.clf()
    plt.cla()
    plt.close()

    plt.title('rank')
    plt.grid(True)
    plt.xlabel('p')
    plt.ylabel('rank')
    plt.plot(PSNR_x, list_k, marker="o")
    plt.xticks(np.arange(0.1, 2.1, 0.1))
    plt.savefig("rank.png")

    """
    クロスバリデーション（1回）
    """
    # クロスバリデーションの繰り返し
    for c in range(1):
        list_MAPE = []
        list_SSIM_closs = []
        list_PSNR_closs = []
        img_learn_array = np.array(img_array)
        # imgをimage_learnとimage_testに分ける
        img_test = np.zeros(int(size*size*0.05))
        img_learn = copy.deepcopy(img)
        # ランダムにx,yを作成してtest_x,test_yに入れる
        # imgのtest_x,test_yの値をimg_testに入れる
        test_x = np.random.randint(0, col - 1, int(size*size*0.05))
        test_y = np.random.randint(0, row - 1, int(size*size*0.05))
        img_test = img[test_y, test_x]
        img_origin = img[test_y, test_x]
        # imgからimg_testが抜けたところを0にしてimg_learnにする
        img_learn[test_y, test_x] = 0
        cv2.imwrite("image_learn_noise.jpg", img_learn * 255)

        for p in np.arange(0.1, 2.1, 0.1):
            # 小数点を調整
            p_str = str('{:.2f}'.format(p))
            f.write('\np=')
            f.write(p_str)
            f.flush()
            # r=差
            r = img_learn_array
            r_new = np.zeros((size, size))
            # M=最終結果
            M = np.zeros((size, size))

            for k in range(40):  # 一番外側　ランク
                # 色々な初期化
                v = copy.deepcopy(v2)
                u = np.zeros((size, 1))
                a = np.zeros((1, size))
                b = np.zeros((1, size))
                c = np.zeros((1, size))
                d = np.zeros((1, size))
                w = np.zeros((1, size))

                E = 50
                print('k_closs=', k)
                for q in range(30):  # 交互反復のループ
                    for i in range(size):  # uの要素数
                        a = []
                        b = []
                        count = 0
                        # ノイズ入ってない部分だけ計算
                        for x in range(size):
                            if img_learn_array[i, x] != 0.1:
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
                            if img_learn_array[i, x] != 0:
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
                    # 多次元を1次元に
                    a_flat = (r - (u @ v)).flatten()
                    b_flat = r.flatten()
                    E_new = (np.linalg.norm(a_flat, ord=2)) ** 2 / (np.linalg.norm(b_flat, ord=2)) ** 2
                    # qの収束条件
                    sigma = np.abs(E - E_new)
                    E = copy.copy(E_new)
                uv = u @ v

                # 思ってたより結果よくならなかったからボツ
                # Mの要素を0~255に絞る（力技）
                # uv[uv < 0] = 0
                # uv[uv > 1] = 1

                r_new = r - uv
                for y in range(size):
                    for x in range(size):
                        if img_learn_array[y, x] == 0:
                            r_new[y, x] = 0
                M += uv
                # kの収束条件
                c_flat = r.flatten()
                d_flat = img_learn.flatten()
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
            # Mの分布図
            """
            plt.scatter(range(22500), M*255, s=3)
            plt.grid()
            plt.savefig(p_str + "M.png")
            # pltの初期化
            plt.clf()
            plt.cla()
            plt.close()
            # Mのup画像
            plt.scatter(range(22500), M * 255, s=3)
            plt.grid()
            plt.ylim(-300, 600)
            plt.savefig(p_str + "M_up.png")
            # pltの初期化
            plt.clf()
            plt.cla()
            plt.close()
            """

            # Mの要素を0~255に絞る（力技）
            M[M < 0] = 0
            M[M > 1] = 1
            # print('max=', np.amax(M))
            # print('min=', np.amin(M))

            M2 = 255*M
            cv2.imwrite(p_str + "_closs.jpg", M2)
            # MAPE出す
            # ここちょっと怪しいかも
            M_test = M[test_y, test_x]
            MAPE = (np.linalg.norm(M_test - img_test, ord=1) / np.linalg.norm(img_test, ord=1))
            print(MAPE)
            list_MAPE.append(MAPE)
            #SSIM_closs
            SSIM_closs = skimage.metrics.structural_similarity(img_test, M_test)
            list_SSIM_closs.append(SSIM_closs)
            #PSNR_closs
            PSNR_closs = cv2.PSNR(img_test, M_test)
            list_PSNR_closs.append(PSNR_closs)

            list_k.append(k)
            k_str = str(k)
            MAPE_str = str(MAPE)
            f.write('\nk_closs=')
            f.write(k_str)
            f.write('\nMAPE=')
            f.write(MAPE_str)
            f.flush()
            """
            クロスバリデーションここまで
            """
        # リストの足し算は＋じゃダメ
        list_MAPE_sum = [MAPE_SUM + MAPE for(MAPE_SUM, MAPE) in zip(list_MAPE_sum, list_MAPE)]
        list_SSIM_closs_sum = [SSIM_SUM + SSIM for(SSIM_SUM, SSIM) in zip(list_SSIM_closs_sum, list_SSIM_closs)]
        list_PSNR_closs_sum = [PSNR_SUM + PSNR for(PSNR_SUM, PSNR) in zip(list_PSNR_closs_sum, list_PSNR_closs)]
        print(len(list_MAPE_sum))
        print(len(list_MAPE))
    # pltの初期化
    plt.clf()
    plt.cla()
    plt.close()

    PSNR_x = np.arange(0.1, 2.1, 0.1)
    # plot_MAPE
    plt.title('MAPE')
    plt.grid(True)
    plt.xlabel('p')
    plt.ylabel('MAPE')
    plt.plot(PSNR_x, list_MAPE, marker="o")
    plt.xticks(np.arange(0.1, 2.1, 0.1))
    plt.savefig("MAPE.png")

    plt.clf()
    plt.cla()
    plt.close()

    # plot_PSNR
    plt.title('PSNR_closs')
    plt.grid(True)
    plt.xlabel('p')
    plt.ylabel('PSNR')
    plt.plot(PSNR_x, list_PSNR_closs, marker="o")
    plt.ylim(10.0, 25.0)
    plt.xticks(np.arange(0.1, 2.1, 0.1))
    plt.savefig("PSNR.png")

    # pltの初期化
    plt.clf()
    plt.cla()
    plt.close()

    # plot_SSIM
    plt.title('SSIM_closs')
    plt.grid(True)
    plt.xlabel('p')
    plt.ylabel('SSIM')
    plt.plot(PSNR_x, list_SSIM_closs, marker="o")
    plt.ylim(0, 1.0)
    plt.xticks(np.arange(0.1, 2.1, 0.1))
    plt.savefig("SSIM.png")

    # pltの初期化
    plt.clf()
    plt.cla()
    plt.close()


