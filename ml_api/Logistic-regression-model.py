#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[2]:


df=pd.read_csv("banking.csv")
df


# In[3]:


df.isnull().sum()


# In[4]:


df = pd.get_dummies(df, drop_first=True)


# In[5]:


X = df.drop('y', axis=1)


# In[6]:


y = df['y']


# In[7]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)




# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


# In[9]:


model = LogisticRegression()
model.fit(X_train, y_train)


# In[10]:

print("X_test value", X_test)
y_pred = model.predict(X_test)


# In[14]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# In[15]:


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# In[16]:


#plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.show()


# In[17]:


from sklearn.utils import resample


# In[18]:


from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


# In[19]:


def evaluate_threshold(y_proba, y_test, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    conf_matrix = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return conf_matrix, precision, recall, f1


# In[20]:


y_proba = model.predict_proba(X_test)[:, 1]


# In[21]:


thresholds = np.linspace(0.2, 0.5, 10)
for threshold in thresholds:
    conf_matrix, precision, recall, f1 = evaluate_threshold(y_proba, y_test, threshold)
    print(f"Threshold: {threshold:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}\n")


# In[22]:


df_majority = df[df.y==0]
df_minority = df[df.y==1]
df_minority_upsampled = resample(df_minority, 
                                 replace=True,    
                                 n_samples=len(df_majority),   
                                 random_state=42) 

df_upsampled = pd.concat([df_majority, df_minority_upsampled])
X_upsampled = df_upsampled.drop('y', axis=1)
y_upsampled = df_upsampled['y']


# In[23]:


X_upsampled_scaled = scaler.fit_transform(X_upsampled)
X_train, X_test, y_train, y_test = train_test_split(X_upsampled_scaled, y_upsampled, test_size=0.3, random_state=42)


# In[24]:


model = LogisticRegression()
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]


# In[25]:


threshold = 0.3 
conf_matrix, precision, recall, f1 = evaluate_threshold(y_proba, y_test, threshold)
print("Confusion Matrix with Resampled Data and New Threshold:")
print(conf_matrix)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")


# In[26]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


# In[27]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[28]:


conf_matrix = confusion_matrix(y_test, (y_proba >= threshold).astype(int))
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted No', 'Predicted Yes'], 
            yticklabels=['Actual No', 'Actual Yes'])
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix')
plt.show()


# In[29]:


fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()


# In[30]:


precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()


# In[31]:


plt.figure(figsize=(8, 6))
plt.plot(pr_thresholds, precision[:-1], 'b--', label='Precision')
plt.plot(pr_thresholds, recall[:-1], 'g-', label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall vs Threshold')
plt.legend(loc='best')
plt.show()


# In[32]:


import pandas as pd


# In[33]:


df = pd.read_csv("banking.csv")


# In[36]:


df['unique_id'] = range(1, len(df) + 1)


# In[37]:


print(df.head())


# In[38]:


import joblib


# In[39]:


joblib.dump(model, 'logistic_regression_model.pkl')

