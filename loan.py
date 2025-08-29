#!/usr/bin/env python
# coding: utf-8

# # LOAN DEFAULT PREDICTION-ML

#     Objective:
#         The Loan Prediction Machine Learning Model is built to assist financial istitutions like banks, NBFC's, microfinance banks in automating the loan approval by predicting whether a loan application should be approved or rejected based on applicant details

# Step 1: Import Dataset

# Import Libraries

# In[1]:


import pandas as pd     # to read dataset
import numpy as np      # to do mathematical operations


# Read Dataset

# In[2]:


df= pd.read_csv("loan_data.csv")        # reads dataset from csv file
df.head()       # shows top 5 rows


#     Numerical:
#         ApplicantIncome	
#         CoapplicantIncome
#         LoanAmount	

#     Categorical:
#         Gender
#         Married
#         Dependents
#         Education
#         Self_Employed
#         Credit_History
#         Property_Area
#         Loan_Status

# Step 2: Data Cleaning and Preprocessing

# In[3]:


df.info()       # shows information about data


# In[4]:


df.isnull().sum()       # shows total null values present in each column


#     Inference:
#         There is a presence of Null values in 'Gender','Dependents','Self Employed', 'Loan Amount Term', 'Credit History' columns
#         These null values has to be either removed or imputed
#         So we are going to impute by idxmax

# In[5]:


df= df.drop(columns=["Loan_ID"])        # to remove particular column from dataset


# In[6]:


df['LoanAmount']=pd.to_numeric(df['LoanAmount'], errors='coerce')       # converting null values of numerical column


# In[7]:


# converting null values by imputing
df['Credit_History'].fillna(df['Credit_History'].value_counts().idxmax(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].value_counts().idxmax(), inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].value_counts().idxmax(), inplace=True)
df['Dependents'].fillna(df['Dependents'].value_counts().idxmax(), inplace=True)
df['Gender'].fillna(df['Gender'].value_counts().idxmax(), inplace=True)


# In[8]:


df.isnull().sum()       # shows total null values in each column


# In[9]:


df.head()       # to show top 5 rows after dropping column


# In[10]:


df.info()       # shows information of dataset


# In[11]:


df.describe()       # describes basic statistical information


# Outliers Detection and Handling

# Import Libraries

# In[12]:


import matplotlib.pyplot as plt     # importing visualization libraries


# Applicant Income

# In[13]:


df.boxplot("ApplicantIncome")       # creates boxplot
plt.title("Applicant Income Outliers")      # shows title of chart
plt.show()      # shows the chart


#     Inference:
#         There is a presence of huge number of outliers in Applicant Income data
#         These outliers are going to removed by Inter Quantile Range (IQR) method

# Applicant Income- outlier removal

# In[14]:


# outliers removed by Inter Quantile Range (IQR) method
Q1= df['ApplicantIncome'].quantile(0.25)
Q3= df['ApplicantIncome'].quantile(0.75)

IQR= Q3-Q1

lower= Q1- 0.5*IQR
upper= Q3-0.5*IQR

df_cleaned= df[(df['ApplicantIncome']>=lower) & (df['ApplicantIncome']<=upper)]


# In[15]:


plt.boxplot(df_cleaned['ApplicantIncome'])      # creates box plot
plt.title("Cleaned Applicant Income")       # shows title of chart
plt.grid()      # shows grid lines
plt.show()      # shows the boxplot chart


#     Inference:
#         Now the outliers are completely removed
#         The Applicant Income Column is purely becomes Normally distributed

# Loan Amount

# In[16]:


df.boxplot("LoanAmount")        # creates boxplot
plt.title("Loan Amount Outliers")       # shows title of chart
plt.show()      # shows entire chart


#     Inference:
#         The Loan Amount has some outliers in it
#         This will affect the prediction of Loan approval status

# Loan Status- Outlier Removal

# In[17]:


# removal of outlier
Q1= df['LoanAmount'].quantile(0.25)
Q3= df['LoanAmount'].quantile(0.75)

IQR= Q3- Q1

lower= Q1-0.5*IQR
upper= Q3-0.5*IQR

df_cleaned= df[(df['LoanAmount']>=lower) & (df['LoanAmount']<=upper)]


# In[18]:


plt.boxplot(df_cleaned['LoanAmount'])       # creates box plot
plt.title("Cleaned Loan Amount")        # shows title of chart
plt.grid()      # shows grid lines in chart
plt.show()      # shows entire graph


#     Inference:
#         The Loan Amount column is completely cleaned
#         This process is done by Inter Quantile Range (IQR) method

# Label Encoding

# In[19]:


from sklearn.preprocessing import LabelEncoder      # import label encoder


# In[20]:


# to do encoding of categorical data
le= LabelEncoder()
for col in ['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area','Loan_Status']:

    df[col]=le.fit_transform(df[col].astype(str))


# In[21]:


df.head()       # shows top 5 rows of cleaned and processed data


# In[22]:


df.info()       # shows information of processed dataset


#     Inference:
#         The dataset is now completey cleaned and encoded for building models

# Step 3: Build Model

#     Inference:
#         The given problem is identified as classification problem.
#         So we are going with Supervised Learning Model with classification models

# Import Libraries

# In[23]:


from sklearn.model_selection import train_test_split        # to split train data and test data
from sklearn.preprocessing import StandardScaler        # importing standard scaler

# supervised learning
from sklearn.neighbors import KNeighborsClassifier      # importing KNeighbours libraray
from sklearn.linear_model import LogisticRegression     # importing linear model library
from sklearn.ensemble import RandomForestClassifier     # importing RandomForest classifier
from sklearn.tree import DecisionTreeClassifier     # importing Decision tree classifier
from xgboost import XGBClassifier     # importing XGBoost Classifier
from sklearn.svm import SVC     # importing Support vector Classifier

from sklearn.metrics import accuracy_score,classification_report, confusion_matrix        # importing Evaluation metrics


# In[24]:


df['Loan_Status'].value_counts()        # shows total counts


#     Inference:
#         The dataset is imbalanced.
#         We need to balance the dataset to avoid bias.
#         So using SMOTE technique

# In[25]:


from imblearn.over_sampling import SMOTE        # to balance the imbalanced data


# Define Features

# In[26]:


X= df[['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']]     # define input data
y= df["Loan_Status"]        # defines target data


# Train-test Split

# In[27]:


X_train, X_test, y_train,y_test= train_test_split(X, y, random_state=42, test_size=0.2)     # train test split in 80:20 ratio


# Scaling and Balancing Dataset

# In[28]:


# scaling the input features
scaler= StandardScaler()

X_train= scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)


# In[29]:


# to balance the imbalanced data
smote= SMOTE()
X_train,y_train= smote.fit_resample(X_train,y_train)


# (1) RandomForestClassifier

# In[30]:


model= RandomForestClassifier()     # select model as RandomForestClassifier
model.fit(X_train,y_train)      # Fit the model


# Prediction

# In[31]:


y_pred_rf= model.predict(X_test)        # Predict the model


# Evaluation

# In[32]:


acc= accuracy_score(y_pred_rf,y_test)       # to evaluate accuracy score
conf_mat= confusion_matrix(y_pred_rf,y_test)        # to calculate confusion matrix
class_rep= classification_report(y_pred_rf,y_test)      # to prepare classification report


# In[33]:


print(f'Accuracy:{acc:.2f}')        # shows accuracy of model
print(f'{conf_mat}')        # shows confusion matrix
print(f'{class_rep}')       # shows classification report


# (2) DecisionTreeClassifier

# In[34]:


model1= DecisionTreeClassifier()        # select model as Decision Tree Classifier
model1.fit(X_train,y_train)     # fit the model


# Prediction

# In[35]:


y_pred_dt= model1.predict(X_test)       # predict the model


# Evaluation

# In[36]:


acc= accuracy_score(y_pred_dt,y_test)       # calculates accuracy score
conf_mat= confusion_matrix(y_pred_dt,y_test)        # prepares confusion matrix
class_rep= classification_report(y_pred_dt,y_test)      # prepares classification report


# In[37]:


print(f'Accuracy:{acc:.2f}')        # shows accuracy score values
print(f'{conf_mat}')        # shows confusion matrix
print(f'{class_rep}')       # shows classification report


# (3) SupporVectorClassifier

# In[38]:


model2= SVC()       # select the Support Vector Classifier
model2.fit(X_train,y_train)     # fit the model


# Prediction

# In[39]:


y_pred_svc= model2.predict(X_test)      # predicts the model


# Evaluation

# In[40]:


acc= accuracy_score(y_pred_svc,y_test)      # calculates accuracy score
conf_mat=confusion_matrix(y_pred_svc,y_test)        # to prepare confusion matrix
class_rep= classification_report(y_pred_svc,y_test)     # to prepare classification report


# In[42]:


print(f'Accuracy:{acc:.2f}')        # shows accuracy score
print(f'{conf_mat}')        # shows confusion matrix
print(f'{class_rep}')       # shows Classification report


# (4) XGBClassifier

# In[43]:


model3= XGBClassifier()     # Select XGBClassifier model
model3.fit(X_train, y_train)        # fit the model


# Prediction

# In[44]:


y_pred_xg= model.predict(X_test)        # predict the model


# Evaluation

# In[45]:


acc= accuracy_score(y_pred_xg,y_test)       # calculates accuracy score
conf_mat= confusion_matrix(y_pred_xg,y_test)        # creates confusion matrix
class_rep= classification_report(y_pred_xg,y_test)      # creates classification report


# In[46]:


print(f'Acuuracy:{acc:.2f}')        # shows accuracy score
print(f'{conf_mat}')        # shows confusion matrix
print(f'{class_rep}')       # shows classification report


# (5) LogisticRegression

# In[47]:


model4= LogisticRegression()        # select Logistic regression model
model4.fit(X_train,y_train)     # fit the model


# Prediction

# In[48]:


y_pred_log= model.predict(X_test)       # predict the model


# Evaluation

# In[49]:


acc= accuracy_score(y_pred_log,y_test)      # calculate accuracy score
conf_mat= confusion_matrix(y_pred_log,y_test)       # prepare confusion matrix
class_rep= classification_report(y_pred_log,y_test)     # prepare classification report


# In[50]:


print(f'Accuracy:{acc:.2f}')        # shows accuracy score value
print(f'{conf_mat}')        # shows confusion matrix
print(f'{class_rep}')       # shows classification result


# (6)KNeighburs

# In[51]:


model5= KNeighborsClassifier()      # select KNeighbour model
model5.fit(X_train,y_train)     # fit the model


# Prediction

# In[52]:


y_pred_kn= model.predict(X_test)        # predict the model


# Evaluation

# In[53]:


acc= accuracy_score(y_pred_kn,y_test)       # calculates accuracy score
conf_mat= confusion_matrix(y_pred_kn,y_test)        # creates confusion matrix
class_rep= classification_report(y_pred_kn,y_test)      # creates classification report


# In[54]:


print(f'Accuracy:{acc:.2f}')        # shows accuracy score value
print(f'{conf_mat}')        # shows confusion matrix
print(f'{class_rep}')       # shows classification report


#     Inference:
#         We built six supervised learning models(RandomForestClassifier, DecisionTreeClassifier,SupportVectorClassifier,XGBoostClassifier, LogisticRegression, KNeighbours), among these six model four models (RandomForestClassifier,XGBoostClassifier, LogisticRegression, KNeighbours) performs well with accuracy score value of 0.81.
#         So we can choose RandomForestClassifier model to Hyperparameter tuning

# Step 4: Hyperparameter Tuning

# Import Library

# In[55]:


from sklearn.model_selection import GridSearchCV        # importing GridsearchCV model


# Tuning- GridSearchCV

# In[56]:


# select parameter for tuning
param_grid={
    'n_estimators': [100,200,300],
    'max_depth':[5,10,None],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,4]
}


# In[57]:


# GridSearch CV
grid_search= GridSearchCV(
    estimator= RandomForestClassifier(random_state=42),
    param_grid= param_grid,
    cv=5,       # 5 cross validation
    n_jobs=-1,
    verbose=1
)


# In[58]:


grid_search.fit(X_train,y_train)        # fit the model


# In[59]:


best_rf= grid_search.best_estimator_


# Prediction

# In[60]:


y_pred_tun= best_rf.predict(X_test)     # predicts the model


# Evaluation

# In[61]:


acc= accuracy_score(y_pred_tun,y_test)      # calculates accuracy score
conf_mat= confusion_matrix(y_pred_tun,y_test)       # creates confusion matrix
class_rep= classification_report(y_pred_tun,y_test)     # creates classification report


# In[62]:


print(f'Accuracy:{acc:.2f}')        # shows accuracy score
print(f'{conf_mat}')        # shows confusion matrix
print(f'{class_rep}')       # shows classification report


# Step 5: Deployment

# Import Library

# In[63]:


import joblib       # import joblib to save model
import streamlit as st      # import streamlit


# In[64]:


model= joblib.dump(model, "loan.pkl")       # save the model


# In[65]:


model6= joblib.load("loan.pkl")      # load the model


# In[66]:


st.title("Loan Prediction")     # create title of page


# to create input features
APP_INC= st.number_input("Applicant Income")
COAPP_INC=st.number_input("Coapplicant Income")
LOAN_TERM= st.selectbox("Loan Term",[12,36,60,84,120,180,240,300,360,380])
LOAN_AMT= st.number_input("Loan Amount")
CRED_HIST= st.selectbox("Credit History",[0,1])
GENDER= st.selectbox("Gender",[0,1])
MARRIED= st.selectbox("Married",[0,1])
DEPENDENT=st.selectbox("Dependent",list(range(0,4)))
Education= st.selectbox("Education",[0,1])
Self_Employed= st.selectbox("Self Employed",[0,1])
Property_Area= st.selectbox("Property Area", [0,1])	



# In[67]:


# to do calculations
input_data= np.array([[APP_INC,COAPP_INC,LOAN_AMT,CRED_HIST,GENDER,MARRIED,LOAN_TERM,DEPENDENT,Education,Self_Employed,Property_Area]])


# predicts result
if st.button("LOAN"):
    prediction= model6.predict(input_data)[0]

    if prediction==1:
        st.success("loan approved")
    else:
        st.error("Loan rejected")
