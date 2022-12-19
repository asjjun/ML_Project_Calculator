from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

star_data = pd.read_csv('star_classification.csv')
star_data = star_data.replace({'class': 'STAR'}, 0)
star_data = star_data.replace({'class': 'GALAXY'}, 1)
star_data = star_data.replace({'class': 'QSO'}, 2)

star_input = star_data[['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']].to_numpy()
star_target = star_data['class'].to_numpy()
star_input = np.delete(star_input, 79543, axis=0)
star_target = np.delete(star_target, 79543, axis=0)

train_input, test_input, train_target, test_target = train_test_split(
    star_input, star_target, random_state=42
)

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

##Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled,train_target)
print("- Train Score : ", dt.score(train_input, train_target))
print("- Test Score : ", dt.score(test_input, test_target))

print("--------Depth [ ", 12, "], Random State [ ", 42, " ]--------")
dt = DecisionTreeClassifier(max_depth=12, random_state=42)
dt.fit(train_input, train_target)
print("- Train Score : ", dt.score(train_input, train_target))
print("- Test Score : ", dt.score(test_input, test_target))

print(dt.feature_importances_)
#depth = 12 random state = 42