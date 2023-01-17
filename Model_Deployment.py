#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import essential libraries

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import joblib


# In[2]:


# Load the daatset
iris_data = load_iris()


# In[3]:


# import dependent and independent data

target = iris_data.target
features = iris_data.data


# In[4]:


target


# In[5]:


features = pd.DataFrame(features,columns=iris_data.feature_names)
features.head()


# In[6]:


scale = StandardScaler()
features_scale = scale.fit_transform(features)


# In[7]:


# Split the dataset in train and test form.
xtrain,xtest,ytrain,ytest = train_test_split(features,target,test_size=0.2,random_state=41)


# In[8]:


xtrain.shape,xtest.shape,ytrain.shape,ytest.shape


# In[9]:


lin_model = LinearRegression()
lin_model.fit(xtrain,ytrain)


# In[10]:


lin_predict = lin_model.predict(xtest)
print(mean_absolute_error(lin_predict,ytest))
print(mean_squared_error(lin_predict,ytest))


# # KNeighborsClassifier

# In[11]:


knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(xtrain,ytrain)


# In[12]:


knn_predict = knn_model.predict(xtest)
accuracy_score(knn_predict,ytest)


# # Support Vector Machine

# In[13]:


svm_model = SVC()
svm_model.fit(xtrain,ytrain)


# In[14]:


svm_predict = svm_model.predict(xtest)
accuracy_score(ytest,svm_predict)


# # Decision Tree 

# In[15]:


tree_model = DecisionTreeClassifier()
tree_model.fit(xtrain,ytrain)


# In[16]:


tree_predict = tree_model.predict(xtest)
accuracy_score(ytest,tree_predict)


# - KNN model has good accuracy.

# # Save the Model

# In[17]:


filename = 'model.pkl'
joblib.dump(knn_model, open(filename, 'wb')) 


# # Deploy the model

# In[18]:


from flask import Flask, request, render_template


# In[19]:


app = Flask(__name__)
model = joblib.load('model.pkl')


# In[20]:


@app.route('/')
# def home():
#     return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():
    int_features = [float(x) for x in request.form.values()]
    print(int_features)
    final_features = [np.array(int_features)]
    print(type(final_features))
    prediction = model.predict(final_features)
    print(type(prediction))
    print(prediction.shape)
    if prediction.item(0) == 0:
        pred_final = "Iris-setosa"
    elif prediction.item(0) == 1:
        pred_final = "Iris-versicolor"
    else:
        pred_final = "Iris-virginica"
    return render_template('index.html', prediction_text=pred_final)


# In[ ]:


if __name__ == '__main__':
    app.run(port=3000,debug=False)


# In[ ]:




