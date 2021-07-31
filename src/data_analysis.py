import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# print(train_data.describe())

# print out number of output for each category
# also plot 
features = train_data.drop(['ID', 'TARGET'], axis=1)
print(train_data['TARGET'].value_counts())
train_data['TARGET'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',shadow=True)
plt.savefig('images/data_analysis.png', bbox_inches='tight')
plt.show()


