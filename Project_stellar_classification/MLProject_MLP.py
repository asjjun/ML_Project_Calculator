import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

star_data = pd.read_csv('star_classification.csv')
star_data = star_data.replace({'class':'STAR'}, 0)
star_data = star_data.replace({'class':'GALAXY'}, 1)
star_data = star_data.replace({'class':'QSO'}, 2)
star_input = star_data[['alpha','delta','u','g','r','i','z','redshift']].to_numpy()
star_target = star_data['class'].to_numpy()
star_input = np.delete(star_input, 79543, axis=0)
star_target = np.delete(star_target, 79543, axis=0)

train_input, test_input, train_target, test_target = train_test_split(
    star_input, star_target, random_state=41)

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(8,)),
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

results = model.fit(train_scaled, train_target, batch_size=1, epochs=20, verbose=1)

evaluate = model.evaluate(test_scaled, test_target, batch_size=1, verbose=2)

model.save('model1.h5')


# model = tf.keras.models.load_model('model1.h5')
predict = model.predict(test_scaled)
print(predict)
