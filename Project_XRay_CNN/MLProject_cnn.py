import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

pixel = 64


def image_save():
    image_arm, image_blade, image_hand = [], [], []
    for i in range(1, 680):
        image_original = cv2.imread("./images/fractured/arm/" + str(i) + ".jpg", cv2.IMREAD_GRAYSCALE)
        image_arm.append(cv2.resize(image_original, (pixel, pixel)))
    for i in range(1, 1274):
        image_original = cv2.imread("./images/not fractured/arm/" + str(i) + ".jpg", cv2.IMREAD_GRAYSCALE)
        image_arm.append(cv2.resize(image_original, (pixel, pixel)))
    for i in range(1, 2003):
        image_original = cv2.imread("./images/fractured/blade/" + str(i) + ".jpg", cv2.IMREAD_GRAYSCALE)
        image_blade.append(cv2.resize(image_original, (pixel, pixel)))
    for i in range(1, 1475):
        image_original = cv2.imread("./images/not fractured/blade/" + str(i) + ".jpg", cv2.IMREAD_GRAYSCALE)
        image_blade.append(cv2.resize(image_original, (pixel, pixel)))
    for i in range(1, 1398):
        image_original = cv2.imread("./images/fractured/hand/" + str(i) + ".jpg", cv2.IMREAD_GRAYSCALE)
        image_hand.append(cv2.resize(image_original, (pixel, pixel)))
    for i in range(1, 1566):
        image_original = cv2.imread("./images/not fractured/hand/" + str(i) + ".jpg", cv2.IMREAD_GRAYSCALE)
        image_hand.append(cv2.resize(image_original, (pixel, pixel)))

    image_arm, image_blade, image_hand = np.array(image_arm), np.array(image_blade), np.array(image_hand)
    np.save('image_arm.npy', image_arm)
    np.save('image_blade.npy', image_blade)
    np.save('image_hand.npy', image_hand)


def image_load():
    image_arm = np.load('image_arm.npy')
    image_blade = np.load('image_blade.npy')
    image_hand = np.load('image_hand.npy')
    return image_arm, image_blade, image_hand


def make_label_part(image_arm, image_blade, image_hand):
    label_arm = [0 for _ in range(len(image_arm))]
    label_blade = [1 for _ in range(len(image_blade))]
    label_hand = [2 for _ in range(len(image_hand))]

    return np.array(label_arm), np.array(label_blade), np.array(label_hand)


def make_label_fracture(image_arm, image_blade, image_hand):
    label_arm = [0 if i < 680 else 1 for i in range(1, len(image_arm) + 1)]
    label_blade = [0 if i < 2003 else 1 for i in range(1, len(image_blade) + 1)]
    label_hand = [0 if i < 1398 else 1 for i in range(1, len(image_hand) + 1)]

    return np.array(label_arm), np.array(label_blade), np.array(label_hand)


def gen_data_part():
    check1 = os.path.isfile(os.getcwd() + '/image_arm.npy')
    check2 = os.path.isfile(os.getcwd() + '/image_blade.npy')
    check3 = os.path.isfile(os.getcwd() + '/image_hand.npy')

    # image_save()는 이미지 파일로 만들어진 npy파일이 없으면 실행
    if not (check1 and check2 and check3):
        image_save()

    # 부위 판별 데이터 생성
    input_arm, input_blade, input_hand = image_load()
    output_arm, output_blade, output_hand = make_label_part(input_arm, input_blade, input_hand)
    image_input = np.concatenate([input_arm, input_blade, input_hand])
    image_output = np.concatenate([output_arm, output_blade, output_hand]).reshape(len(image_input), 1)
    # 부위 판별 데이터의 골절 여부 label
    fracture_a, fracture_b, fracture_h = make_label_fracture(input_arm, input_blade, input_hand)
    fracture_target = np.concatenate([fracture_a, fracture_b, fracture_h]).reshape(len(fracture_a) + len(fracture_b) + len(fracture_h), 1)

    # 정규화
    train_scaled = image_input.reshape(-1, pixel, pixel, 1) / 255.0

    # shuffle & split
    train_input, test_input, train_target, test_target = train_test_split(
        train_scaled, image_output, test_size=0.25, random_state=41)

    _, _, _, fracture_label_ = train_test_split(
        train_scaled, fracture_target, test_size=0.25, random_state=41)
    return train_input, test_input, train_target, test_target, fracture_label_


def gen_data_fracture():
    # 골절 판별 데이터 생성
    input_arm, input_blade, input_hand = image_load()
    output_arm, output_blade, output_hand = make_label_fracture(input_arm, input_blade, input_hand)

    # 정규화
    input_arm_scaled = input_arm.reshape(-1, pixel, pixel, 1) / 255.0
    input_blade_scaled = input_blade.reshape(-1, pixel, pixel, 1) / 255.0
    input_hand_scaled = input_hand.reshape(-1, pixel, pixel, 1) / 255.0

    # shuffle & split
    train_input_a, test_input_a, train_target_a, test_target_a = train_test_split(
        input_arm_scaled, output_arm, test_size=0.2, random_state=41)

    train_input_b, test_input_b, train_target_b, test_target_b = train_test_split(
        input_blade_scaled, output_blade, test_size=0.2, random_state=41)

    train_input_h, test_input_h, train_target_h, test_target_h = train_test_split(
        input_hand_scaled, output_hand, test_size=0.2, random_state=41)

    return train_input_a, test_input_a, train_target_a, test_target_a, \
           train_input_b, test_input_b, train_target_b, test_target_b, \
           train_input_h, test_input_h, train_target_h, test_target_h


def model_fit_part(train_input, test_input, train_target, test_target):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(pixel, pixel, 1)))
    model.add(keras.layers.MaxPooling2D(2))

    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D(2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(3, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_input, train_target, epochs=20, verbose=1)

    model.save('model_part.h5')

    model.evaluate(test_input, test_target)


def model_fit_fracture(train_input, test_input, train_target, test_target, part):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(pixel, pixel, 1)))
    model.add(keras.layers.MaxPooling2D(2))

    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D(2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_input, train_target, epochs=20, verbose=1)

    model.save('model_fracture_'+part+'.h5')

    model.evaluate(test_input, test_target)


def model_predict(test_input, test_target, fract_label):
    model = tf.keras.models.load_model('model_part.h5')
    model_arm = tf.keras.models.load_model('model_fracture_arm.h5')
    model_blade = tf.keras.models.load_model('model_fracture_blade.h5')
    model_hand = tf.keras.models.load_model('model_fracture_hand.h5')

    predict = model.predict(test_input)

    predict = np.argmax(predict, axis=1).astype(int).reshape(len(predict), 1)
    # print("predict\n", predict)
    # print("test_target\b", test_target)
    print("1. 부위 판별")
    print("original label\n", test_target)
    print("predict label\n", predict)

    error = list(np.equal(test_target, predict)).count(False)
    print("test data, error data = %d, %d" % (len(predict), error))
    print("MAE: ", error / len(predict))

    result = []
    for i in range(len(predict)):
        if predict[i] == 0:
            tmp = model_arm.predict(test_input[i].reshape(-1, pixel, pixel, 1))
            result.append(round(tmp[0][0]))
        elif predict[i] == 1:
            tmp = model_blade.predict(test_input[i].reshape(-1, pixel, pixel, 1))
            result.append(round(tmp[0][0]))
        elif predict[i] == 2:
            tmp = model_hand.predict(test_input[i].reshape(-1, pixel, pixel, 1))
            result.append(round(tmp[0][0]))
    result = np.array(result).reshape(len(result), 1)

    print("2. 골절 판별")
    print("original label\n", fract_label)
    print("predict label\n", result)

    error = list(np.equal(fract_label, result)).count(False)
    print("test data, error data = %d, %d" % (len(fract_label), error))
    print("MAE: ", error / len(fract_label))


def maincode():
    # 골절 판별 데이터 생성
    train_inp_a, test_inp_a, train_tar_a, test_tar_a, \
    train_inp_b, test_inp_b, train_tar_b, test_tar_b, \
    train_inp_h, test_inp_h, train_tar_h, test_tar_h = gen_data_fracture()

    # 골절 판별 모델 학습은 학습된 모델파일(.h5)이 없으면 실행
    check1 = os.path.isfile(os.getcwd() + '/model_fracture_arm.h5')
    check2 = os.path.isfile(os.getcwd() + '/model_fracture_blade.h5')
    check3 = os.path.isfile(os.getcwd() + '/model_fracture_hand.h5')
    if not (check1 and check2 and check3):
        # 골절 판별 모델 학습
        model_fit_fracture(train_inp_a, test_inp_a, train_tar_a, test_tar_a, "arm")
        model_fit_fracture(train_inp_b, test_inp_b, train_tar_b, test_tar_b, "blade")
        model_fit_fracture(train_inp_h, test_inp_h, train_tar_h, test_tar_h, "hand")

    # 부위 판별 데이터 생성
    train_inp, test_inp, train_tar, test_tar, fracture_label = gen_data_part()

    # 부위 판별 모델 학습은 학습된 모델파일(.h5)이 없으면 실행
    check1 = os.path.isfile(os.getcwd() + '/model_part.h5')
    if not check1:
        # 부위 판별 학습
        model_fit_part(train_inp, test_inp, train_tar, test_tar)

    # 부위 예측 및 골절 예측
    model_predict(test_inp, test_tar, fracture_label)


maincode()


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
각 단계 별 따로 진행 하고 싶을때 주석 처리하며 실행 할 수 있는 코드
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# # 골절 판별 데이터 생성
# train_inp_a, test_inp_a, train_tar_a, test_tar_a, \
# train_inp_b, test_inp_b, train_tar_b, test_tar_b, \
# train_inp_h, test_inp_h, train_tar_h, test_tar_h = gen_data_fracture()
#
# # 골절 판별 모델 학습
# model_fit_fracture(train_inp_a, test_inp_a, train_tar_a, test_tar_a, "arm")
# model_fit_fracture(train_inp_b, test_inp_b, train_tar_b, test_tar_b, "blade")
# model_fit_fracture(train_inp_h, test_inp_h, train_tar_h, test_tar_h, "hand")

# # 부위 판별 데이터 생성
# train_inp, test_inp, train_tar, test_tar, fracture_label = gen_data_part()
# #
# # # 부위 판별 학습
# # model_fit_part(train_inp, test_inp, train_tar, test_tar)
# # #
# # 부위 예측 및 골절 예측
# model_predict(test_inp, test_tar, fracture_label)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
