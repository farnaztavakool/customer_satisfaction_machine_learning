import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Exploratory Data Analysis for Santander Bank

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

df_train = df_train.drop(columns = "ID")
df_test = df_test.drop(columns = "ID")
y_train = df_train['TARGET'].copy()


# Plotting the distribution of Target = 1 and 0

def target_plot(df,h=500,size=(8,8)):

    plt.figure(figsize=size)
    ax = sns.countplot(x=y_train,data=df_train, palette= ["#7fcdbb", "#edf8b1"])
    total = df.shape[0]
    # adapted from https://stackoverflow.com/questions/31749448/how-to-add-percentages-on-top-of-bars-in-seaborn
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + h,
                '{:1.2f}%'.format(height*100/total),
                ha="center") 
    plt.title("Target Feature Distribution")
    df[df['TARGET']==1].shape[0]
    df[df['TARGET']==0].shape[0]
    plt.savefig('target_dist.png')
    
y_col = 'TARGET'
target_plot(df_train)


# var3 exploration
var3 = df_train['var3']
l = []
for i in range(len(var3)):
    if var3[i] == 2:
        l.append(i)
val3_2 = len(l)

# var3 distribution histogram
plt.hist(df_train['var3'], bins=50, range=(5, 300), alpha=0.7, color='#896279')
plt.xlabel('var3')
plt.title('var3 distribution')
plt.savefig('var3.png')

# var15 exploration
max_var15 = df_train['var15'].max()
min_var15 = df_train['var15'].min()


# var15 distribution
unsat = df_train[(df_train['TARGET']==1)]
plt.hist(unsat['var15'],color='#9DCED2')
plt.title("var15 where customers are unsatisfied")
plt.xlim(15, 200)
plt.xlabel('var15')
plt.savefig('var15.png')

min_unsat_age = unsat['var15'].min()
max_unsat_age = unsat['var15'].max()


# train data dist var15
df_train['var15'].hist(bins=100, range=(0, 1000), alpha=0.5, color='#D11149')
plt.xlabel('var15')
plt.xlim(0, 150)     
plt.title('var15 distribution over train data')
plt.savefig('var15_train.png')
    

# test data dist var15
df_test['var15'].hist(bins=100, range=(0, 1000), alpha=0.5, color='#82D4BB')
plt.xlabel('var15')
plt.xlim(0, 150) 
plt.title('var15 distribution over test data')
plt.savefig('var15_test.png')

# var21 exploration
min_var21 = df_train['var21'].min()
max_var21 = df_train['var21'].max()
unique_var21 = df_train['var21'].unique()


# var 21 distribution
df_train['var21'].hist(bins=100, range=(0, 30000), alpha=0.7, color='#D11149')
plt.xlabel('var21')
plt.title('var21 distribution')
plt.xlim(300, 30000) 
plt.ylim(0,250)
plt.savefig('var21.png')

# var36 exploration
min_var36 = df_train['var36'].min()
max_var36 = df_train['var36'].max()
unique_var36 = df_train['var36'].unique()

# var36 distribution
df_train['var36'].hist(bins=100, figsize=(10, 8), alpha=0.5, color='#36C9C6')
plt.xlabel('var36')
plt.title('var36 distribution')
plt.savefig('var36.png')

# var38 exploration
min_var38 = df_train['var38'].min()
max_var38 = df_train['var38'].max()

# var38 distribution
df_train['var38'].hist(bins=100, range=(0, 500000), alpha=0.7, color='#D11149')
plt.xlabel('var38')
plt.title('var38 distribution')
plt.savefig('var38.png')


# Correlation of top 10 features with target
#Adapted from https://www.kaggle.com/saga21/customer-satisfaction-models-explainability/notebook
correlation = df_train.corr()
highly_correlated = correlation.nlargest(10, 'TARGET')['TARGET']
fig, ax = plt.subplots(1, 1, figsize=(7,4))
plt.bar(highly_correlated[1:].index.values, highly_correlated[1:].values, alpha=0.7, color='#D1D646')
plt.title("highly_correlated features with TARGET")
plt.ylabel("Correlation")
plt.xlabel("Features")
plt.xticks(rotation=90)
plt.savefig('correlation_with_target.png')


