import random
import numpy as np
import tensorflow as tf
from numpy import log as ln
np.set_printoptions(suppress=True)


# 데이터 생성
def gen_data(n_examples, n_numbers, max_value):
    input_value, output_value = [], []
    for _ in range(n_examples):
        # n_numbers = 2면 [x1, x2] 형태의 데이터 생성
        x = [random.random() * max_value for _ in range(n_numbers)]
        y = x[0]
        # n_numbers값에 따라 for문이 돌면서 뒤에 입력한 값들 계산
        for j in range(1, n_numbers):
            if symbol == '+':
                y = y + x[j]
            elif symbol == '-':
                y = y - x[j]
            elif symbol == '*':
                y = y * x[j]
            elif symbol == '/':
                y = y / x[j]
        input_value.append(x)
        output_value.append(y)

    # numpy 배열로 변환
    input_value = np.array(input_value)
    output_value = np.array(output_value).reshape(n_examples, 1)
    return input_value, output_value


#  데이터 전처리 및 정규화
def preprocessing(x, y, max_value, n_numbers):
    x_processed, y_processed = None, None
    if symbol == '+':  # '+'는 더해서 나올 수 있는 최댓값(100+100)으로 나눠줌
        x_processed = x / (max_value * n_numbers)
        y_processed = y / (max_value * n_numbers)
    elif symbol == '-':  # '-'는 음수 방지로 max_value를 더 해준 후 더해서 나올 수 있는 최댓값으로 나눠줌
        x_processed = (x + max_value) / (max_value * n_numbers)
        y_processed = (y + max_value) / (max_value * n_numbers)
    elif symbol == '*':  # '*'는 자연로그 ln의 성질을 이용해서 ln을 씌워주고, 0~1사이로 만들기위해 10으로 나눔
        x_processed = ln(x) / 10
        y_processed = ln(y) / 10
    elif symbol == '/':  # '/'는 곱셈과 같지만 음수 방지를 위해 +5 해줌
        x_processed = (ln(x) + 5) / 10
        y_processed = (ln(y) + 5) / 10
    return x_processed, y_processed


# 원래 데이터로 복구
def restore_data(data, max_value, n_numbers):
    restored_data = None
    if symbol == '+':
        restored_data = data * (max_value * n_numbers)
    elif symbol == '-':
        restored_data = data * (max_value * n_numbers) - max_value
    elif symbol == '*':
        restored_data = np.exp(data * 10)
    elif symbol == '/':
        restored_data = np.exp(data * 10 - 5)
    return restored_data


# 모델 구성
def model_config():
    model_tmp = None
    if symbol == '+':
        model_tmp = tf.keras.Sequential([
            tf.keras.layers.Dense(num_numbers*3, activation='relu', input_shape=(num_numbers,)),
            tf.keras.layers.Dense(num_numbers*2, activation='relu'),
            tf.keras.layers.Dense(1, activation='relu')
        ])
    elif symbol == '-':
        model_tmp = tf.keras.Sequential([
            tf.keras.layers.Dense(num_numbers*2, activation='relu', input_shape=(num_numbers,)),
            tf.keras.layers.Dense(num_numbers, activation='relu'),
            tf.keras.layers.Dense(1, activation='relu')
        ])
    elif symbol == '*':
        model_tmp = tf.keras.Sequential([
            tf.keras.layers.Dense(num_numbers * 2, activation='relu', input_shape=(num_numbers,)),
            tf.keras.layers.Dense(num_numbers, activation='relu'),
            tf.keras.layers.Dense(1, activation='relu')
        ])
    elif symbol == '/':
        model_tmp = tf.keras.Sequential([
            tf.keras.layers.Dense(num_numbers * 2, activation='relu', input_shape=(num_numbers,)),
            tf.keras.layers.Dense(num_numbers, activation='relu'),
            tf.keras.layers.Dense(1, activation='relu')
        ])
    return model_tmp


# 데이터 값 설정
symbol = '+'
num_numbers = 2  # 2로 하면 2개의 숫자를 입력해서 계산
num_examples = 5000  # 총 문제 개수
test_num_examples = 1000  # 테스트 문제 개수
largest = 100  # 입력 숫자의 최댓값 설정

# 학습 데이터
train_x, train_y = gen_data(num_examples, num_numbers, largest)
train_input, train_output = preprocessing(train_x, train_y, largest, num_numbers)
print("======= input 데이터 =======\n", train_x)
print("======= output 데이터 =======\n", train_y)
# 테스트 데이터
test_x, test_y = gen_data(test_num_examples, num_numbers, largest)
test_input, test_output = preprocessing(test_x, test_y, largest, num_numbers)


model = model_config()

model.compile(optimizer='adam', loss='mse')

results = model.fit(train_input, train_output, batch_size=1, epochs=20, verbose=1)

evaluate = model.evaluate(test_input, test_output, batch_size=1, verbose=2)

predict = model.predict(test_input)
predict_data = restore_data(predict, largest, num_numbers)

# 예측 값 확인
for i in range(20):
    print("\ninput data: ", test_x[i])
    print('Original, Predicted = %.3f, %.3f (err=%.3f)' % (test_y[i], predict_data[i], test_y[i] - predict_data[i]))

# 평균 절대 오차
error_data = np.abs(test_y - predict_data)
mae = sum(error_data) / num_examples
print("\n\nMAE: %.20f" % mae[0])
