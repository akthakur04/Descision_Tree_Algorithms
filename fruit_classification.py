
#import data files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


#load data set
url="fruit_data_with_colours.csv"
fruits=pd.read_csv(url)

#get info
print(fruits.head(20))
print(fruits.info())
print(fruits.shape)
print(fruits['fruit_name'].unique())


#data visualization
#countplot
sns.countplot(fruits['fruit_name'],label='count')
plt.show()
#box plot
fruits.drop('fruit_label',axis=1).plot(kind='box',subplots=True,
                                       layout=(2,2),sharex=True,sharey=False,figsize=(9,9),title='BoxPLot')
plt.savefig('fruit_box')
plt.show()


#histogram
fruits.drop('fruit_label',axis=1).hist(bins=30,figsize=(9,9))
pl.suptitle('Histogram for each numeric input')
plt.savefig("fruits_hist")
plt.show()



#train and test the data set
featue_names=['mass','width','height','color_score']
x=fruits[featue_names]
y=fruits['fruit_label']

x_train,x_test,y_train,y_test=tts(x,y,random_state=0)

#normalize data
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)


clf=DecisionTreeClassifier()
f=clf.fit(x_train,y_train)

#check accuracy
print("Accuracy of Descison Tree Classifier on training set is  {:.2f}".format(f.score(x_train,y_train)))
print("Accuracy of Descison Tree Classifier on testing set is  {:.2f}".format(f.score(x_test,y_test)))
predict=f.predict(x_test)
print("\n",predict)
df=pd.DataFrame({"actual":y_test,"predicted":predict})
print(df)

#plot decision tree
tree.plot_tree(f)

plt.show()
