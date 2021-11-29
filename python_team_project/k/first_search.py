import networkx as nx
import osmnx as ox
import cv2
import numpy as np
from tqdm import tqdm
# import SearchPathAlgorithm as al
import random
import os
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import pandas as pd
import shutil
from keras.models import load_model
from k import SearchPathAlgorithm as al


def search(time, start, end, route):
    s_node = 0
    e_node = 0
    print('메인 지도 그리는 중')
    G = ox.graph_from_point(start, dist=2500, dist_type='bbox', network_type='walk')
    print('경로 그리는 중')
    fig, ax = ox.plot_graph_route(G, route, node_size=0, route_linewidth=10, show=False)
    print('배경과 경로 분리 중')
    img = node_division(fig)

    print('기존 사진 삭제 중')
    file_setting()
    #
    g_node = len(list(G))
    # #
    wrong = 0
    print('opencv 1:1 비교 시작')
    for i in tqdm(range(50)):
        while True:
            g_node_range = random.randrange(1, 8)
            route_list = [list(G)[random.randrange(0, g_node - 1)] for i in range(g_node_range)]
            length_sum, time_sum, success, route_sum, start_node, end_node = al.route(route_list, G, start, end,time)
            s_node = start_node
            e_node = end_node
            print(time_sum)
            if success:  # 만약 성공시
                set_route = set(route_sum)  # 너무 중복된 길인지 확인
                if len(set_route) / len(route_sum) > 0.85:
                    fig, ax = ox.plot_graph_route(G, route_sum, node_size=0, route_linewidth=10, show=False)
                    img_compare = node_division(fig)
                    dst = image_compare(img, img_compare)

                    if dst / 256 < 0.01:
                        print('#############################################')
                        cv2.imwrite(os.path.dirname(os.path.abspath(__file__))+'/Photo/answer/' + str(i) + '.png', img_compare)
                        break
                    else:
                        if wrong < 50:
                            print('---------------------------------------------')
                            cv2.imwrite(os.path.dirname(os.path.abspath(__file__))+'/Photo/wrong/' + str(wrong) + '.png', img_compare)
                            wrong += 1
    #
    # # ###############################################################################################
    # # # 텐서 시작
    print('텐서로 학습 시작')
    # start_node = ox.nearest_nodes(G, start[0], start[1])
    # end_node = ox.nearest_nodes(G, end[0], end[1])
    print('파일 경로 생성')
    file_path = str(s_node) + '_' + str(e_node)
    #
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.dirname(os.path.abspath(__file__))+'/Photo/',
        image_size=(64, 64),
        batch_size=10,
        subset='training',
        validation_split=0.2,
        seed=1234
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.dirname(os.path.abspath(__file__))+'/Photo/',
        image_size=(64, 64),
        batch_size=10,
        subset='validation',
        validation_split=0.2,
        seed=1234
    )

    train_ds = train_ds.map(fff)
    val_ds = val_ds.map(fff)

    if not os.path.isdir(os.path.dirname(os.path.abspath(__file__))+'/model/' + file_path):
        model = tf.keras.Sequential([

            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu', input_shape=(64, 64, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])

        model.summary()

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model = load_model(os.path.dirname(os.path.abspath(__file__))+'/model/' + file_path)

    model.fit(train_ds, validation_data=val_ds, epochs=100)

    model.save(os.path.dirname(os.path.abspath(__file__))+'/model/' + file_path)
    ##############################################################################################
    # 텐서로 데이터 수집
    # model = load_model('./model/' + file_path)
    print('텐서로 좀 더 나은 데이터 수집')
    node_data(file_path)
    df = pd.read_csv(os.path.dirname(os.path.abspath(__file__))+'/data/' + file_path + '.csv')

    print('데이터 수집 시작')
    for i in tqdm(range(500)):
        while True:
            g_node_range = random.randrange(1, 8)
            route_list = [list(G)[random.randrange(0, g_node - 1)] for i in range(g_node_range)]
            length_sum, time_sum, success, route_sum, start_node, end_node = al.route(route_list, G, start, end,
                                                                                      time)
            if success:  # 만약 성공시
                set_route = set(route_sum)  # 너무 중복된 길인지 확인
                if len(set_route) / len(route_sum) > 0.85:
                    fig, ax = ox.plot_graph_route(G, route_sum, node_size=0, route_linewidth=10, show=False)
                    img_compare = node_division(fig)

                    img_compare = cv2.resize(img_compare, (64, 64))
                    img_array = image.img_to_array(img_compare)
                    img_batch = np.expand_dims(img_array, axis=0)
                    prediction = model.predict(img_batch)
                    if int(prediction[0]) == 0:
                        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                        new_data = {
                            'time' : time,
                            'node' : route_sum,
                            'start' : start_node,
                            'end' : end_node
                        }
                        df = df.append(new_data, ignore_index=True)
                        print(df)
                        # df.to_csv('./data/' + file_path + '.csv', index=False)
                        break
    df.to_csv(os.path.dirname(os.path.abspath(__file__))+'/data/' + file_path + '.csv', index=False)


def node_data(file_path):
    if not os.path.isfile(os.path.dirname(os.path.abspath(__file__))+'/data/' + file_path + '.csv'):
        df = pd.DataFrame(columns=['time', 'node', 'start', 'end'])
        df.to_csv(os.path.dirname(os.path.abspath(__file__))+'/data/' + file_path + '.csv', index=False)


def file_setting():
    shutil.rmtree(os.path.dirname(os.path.abspath(__file__))+'/Photo/answer')
    shutil.rmtree(os.path.dirname(os.path.abspath(__file__))+'/Photo/wrong')
    os.makedirs(os.path.dirname(os.path.abspath(__file__))+'/Photo/answer')
    os.makedirs(os.path.dirname(os.path.abspath(__file__))+'/Photo/wrong')


def node_division(fig):
    # 맵 데이터인 fig를 이미지화 시킴
    # 그러면서 빨간색 경로가 파랑색으로 바뀜
    img_color = np.array(fig.canvas.renderer._renderer)

    height, width = img_color.shape[:2]  # 이미지의 높이와 너비 불러옴, 가로 [0], 세로[1]

    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)  # cvtColor 함수를 이용하여 hsv 색공간으로 변환

    # 추출할 파랑색의 범위를 지정하는 부분
    lower_blue = (120 - 10, 30, 30)  # hsv 이미지에서 바이너리 이미지로 생성 , 적당한 값 30
    upper_blue = (120 + 10, 255, 255)

    img = cv2.inRange(img_hsv, lower_blue, upper_blue)  # 범위내의 픽셀들은 흰색, 나머지 검은색

    # 변환된 mask 이미지를 보통 이미지로 전환
    img_mask_compare_0 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img_mask_compare_0


def image_compare(img_mask_compare_0, img_mask_compare_1):
    # 이미지를 16x16 크기의 평균 해쉬로 변환 ---②
    def img2hash(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (16, 16))
        avg = gray.mean()
        bi = 1 * (gray > avg)
        return bi

    # 해밍거리 측정 함수 ---③
    def hamming_distance(a, b):
        a = a.reshape(1, -1)
        b = b.reshape(1, -1)
        # 같은 자리의 값이 서로 다른 것들의 합
        distance = (a != b).sum()
        return distance

    # 권총 영상의 해쉬 구하기 ---④
    query_hash = img2hash(img_mask_compare_0)

    # 데이타 셋 영상 한개의 해시  ---⑦
    a_hash = img2hash(img_mask_compare_1)
    # 해밍 거리 산출 ---⑧
    dst = hamming_distance(query_hash, a_hash)

    return dst


def fff(i, result):
    i = tf.cast(i / 255.0, tf.float32)
    return i, result


if __name__ == '__main__':
    print('초기 셋팅 시작')
    time = 60
    start = (35.8956224, 128.6224266)
    end = (35.888836, 128.6102997)
    # route = [5540134344, 4634458114, 3968872959, 6077982441, 6077982439, 6077982438, 6077982448, 6077982447, 7712664005,
    #          6077982445, 6077982443, 436847411, 3969143356, 436847410, 6110920583, 3969015056, 4864256203, 4864256198,
    #          4864256205, 8247026945, 5343005483, 8247026977, 8247026978, 697232798, 5343005479, 369914144, 8246998080,
    #          4641802035, 7098099026, 4079631406, 697232797, 5343005470, 4641802061, 3969143335, 4079631475, 4079631487,
    #          5551904035, 4079631506, 4079631499, 420450813, 420450808, 5100101377, 420450800, 5551903411, 420450804,
    #          5608611406, 5608611415, 420450805, 5608611402, 436816081, 5624709473, 5635283399, 5101432040, 5101432041,
    #          5101432039, 5101432037, 5124316811, 5104520888, 5578497792, 5104520893, 7966922418, 4631924213, 5608591572,
    #          7966949127, 5091020036, 5088768411, 7966907612, 5088812286, 4641219047, 5647912227]
    route = [5540134344, 4634458114, 6081123537, 4634458115, 5095171222, 436816090, 3969143758, 4636709801, 3969143355, 4636709805, 7855930795, 5551875821, 3969143347, 8155298669, 3969143337, 3969143335, 4079631475, 4079631476, 4079631469, 4079631463, 4079631467, 4079631464, 4079631465, 4079631451, 4079631461, 4079631452, 4079631458, 4079631454, 4079631456, 4079631233, 4079631231, 4079631191, 4079631250, 4079631272, 420450781, 436843739, 420450813, 420450808, 5100101377, 420450800, 5551903411, 420450804, 5608611406, 5608611415, 420450805, 5608611402, 436816081, 5624709473, 5635283399, 5101432040, 5101432041, 7966908245, 7966908252, 7966908238, 7966908237, 5578497789, 5104520886, 5104520896, 4631924184, 5102780795, 4641219050, 5104520884, 4641219048, 5124032461]

    print('셋팅 끝')
    search(time, start, end, route)
