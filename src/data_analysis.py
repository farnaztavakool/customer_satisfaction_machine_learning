import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif, VarianceThreshold



train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# print(train_data.describe())

# print out number of output for each category
# also plot 
features = train_data.drop(['ID', 'TARGET'], axis=1)
y = train_data['TARGET']
print(train_data['TARGET'].value_counts())
train_data['TARGET'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',shadow=True)
plt.savefig('images/data_analysis.png', bbox_inches='tight')
plt.show()

# drop constant features
constant_filter = VarianceThreshold(threshold= 0.001)
features = pd.DataFrame(constant_filter.fit_transform(features))

f_score = f_classif(features, y)

p_values=pd.Series(f_score[0])

p_values.sort_values(ascending=False).plot.bar(figsize=(20, 8))
plt.show()




