# -*-coding:utf8-*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

import data_path as files

data = pd.read_csv(files.train_file)
print(data.info())
'''<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object'''

# 性别男的值为1，女为0
data['Sex'] = data['Sex'].apply(lambda s:1 if s == 'male' else 0)

mean_age = data['Age'].mean()
data['Age'][data.Age.isnull()] = mean_age

# 缺失值为-1
data = data.fillna(0)

# 取Sex,Age,Pclass,SibSp,Parch,Fare分析
dataset_X = data[['Sex','Age','Pclass','SibSp','Parch','Fare']]

dataset_X = dataset_X.as_matrix()

# print(dataset_X[888])
# deceased status
data['Deceased'] = data["Survived"].apply(lambda s:int(not s))

dataset_Y = data[['Survived','Deceased']]
dataset_Y = dataset_Y.as_matrix()

# print(dataset_Y)

X_train,X_val,Y_train,Y_val = train_test_split(
    dataset_X,dataset_Y,test_size=0.2,random_state=42
)

# print(len(X_train))
X = tf.placeholder(tf.float32,shape=[None,6])
y = tf.placeholder(tf.float32,shape=[None,2])

# 设置权重W和偏置向量b
W = tf.Variable(tf.random_normal([6,2]),name='weight')
b = tf.Variable(tf.zeros([2]),name='bias')

y_pred = tf.nn.softmax(tf.matmul(X,W)+b)

cross_entropy = - tf.reduce_sum(y * tf.log(y_pred + 1e-10),
                                reduction_indices=1)
cost = tf.reduce_mean(cross_entropy)

train_op = tf.train.GradientDescentOptimizer(1e-3).minimize(cost)

# calculate accuracy
correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
acc_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    # init all variables
    tf.global_variables_initializer().run()

    for epoch in range(10):
        total_loss = 0
        for i in range(len(X_train)):
            feed = {X:[X_train[i]],y:[Y_train[i]]}
            _,loss = sess.run([train_op,cost],feed_dict=feed)
            total_loss += loss
        print('Epoch: %04d, total loss=%.9f' % (epoch+1,total_loss))
    print('Training complete')
    # Accuracy calculated by TensorFlow
    accuracy = sess.run(acc_op, feed_dict={X: X_val, y: Y_val})
    print("Accuracy on validation set: %.9f" % accuracy)

    # Accuracy calculated by NumPy
    pred = sess.run(y_pred, feed_dict={X: X_val})
    correct = np.equal(np.argmax(pred, 1), np.argmax(Y_val, 1))
    numpy_accuracy = np.mean(correct.astype(np.float32))
    print("Accuracy on validation set (numpy): %.9f" % numpy_accuracy)

# predict on test data
    testdata = pd.read_csv(files.test_file)
    testdata = testdata.fillna(-1)
    # convert ['male', 'female'] values of Sex to [1, 0]
    testdata['Sex'] = testdata['Sex'].apply(lambda s: 1 if s == 'male' else 0)
    X_test = testdata[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
    predictions = np.argmax(sess.run(y_pred, feed_dict={X: X_test}), 1)
    submission = pd.DataFrame({
        "PassengerId": testdata["PassengerId"],
        "Survived": predictions
    })

    submission.to_csv(files.data_path+"titanic-submission.csv", index=False)