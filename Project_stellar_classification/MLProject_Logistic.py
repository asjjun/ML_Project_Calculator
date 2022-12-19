from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

star_data = pd.read_csv('star_classification.csv')
star_data = star_data.replace({'class':'STAR'}, 0)
star_data = star_data.replace({'class':'GALAXY'}, 1)
star_data = star_data.replace({'class':'QSO'}, 2)

star_input = star_data[['alpha','delta','u','g','r','i','z','redshift']].to_numpy()
star_target = star_data['class'].to_numpy()
star_input = np.delete(star_input, 79543, axis=0)
star_target = np.delete(star_target, 79543, axis=0)

train_input, test_input, train_target, test_target = train_test_split(
        star_input, star_target, random_state=42
)

ss= StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

##Logistic
lr = LogisticRegression(C=30,max_iter=100000)
lr.fit(train_scaled, train_target)
print("Logistic")
print("- Train Score : ", lr.score(train_scaled, train_target))
print("- Test Score : ", lr.score(test_scaled, test_target))
print(lr.coef_,lr.intercept_)

