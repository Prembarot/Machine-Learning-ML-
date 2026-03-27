from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sn 

### DATA PREPROCESSING - STARTS
digits=load_digits()
# print(dir(digits))

# plt.gray()
# for i in range(10):   
#     plt.matshow(digits.images[i])
# plt.show()

# print(digits.DESCR)
# print(digits.data[:5])

df=pd.DataFrame(digits.data)
# print(df.head())                    ## 8 x 8 ==> 64    FEATURES
# print(digits.target)                  ## LABELS 
## ORIGINAL DF SHAPE ==> (1797,64)

df["target"] = digits.target
# print(df.shape)                      ## (1797, 65) after adding target in df ==> 65
# print(df.head())
X = df.drop("target", axis=1)
y = df[["target"]]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

# print("X_train shape:", X_train.shape)          ## (1437, 64)
# print("y_train shape:", y_train.shape)          ## (1437, 1)
# print("X_test shape :", X_test.shape)           ## (360, 64)
# print("y_test shape :", y_test.shape)           ## (360, 1)

### DATA PREPROCESSING - ENDS

### TRAINING MODEL - STARTS
model= RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)
print("Model score :",model.score(X_test,y_test))                ## 0.9805555555555555
# print("ACTUAL VALUE :")
# print(y_test)
# print("PRICRED VALUE :")
y_pred=model.predict(X_test)
# print(y_pred)
COMMAT=confusion_matrix(y_test,y_pred)
print("CONFUSION MATRIX :",COMMAT)
plt.figure(figsize=(10,7))
sn.heatmap(COMMAT,annot=True,cmap='Reds')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


### TRAINING MODEL - ENDS

# y_pred_train= model.predict(X_train)
# # print(y_pred_train)
# y_pred_test= model.predict(X_test)
# # print(y_pred_test)
# print("DECISION TREE SCORE")
# print("Training Score :",model.score(X_train,y_pred_train))
# print("Testing Score :",model.score(X_test,y_pred_test))
# print("DECISION TREE ACCURACY SCORE")
# print("Training accuracy :",accuracy_score(y_train,y_pred_train))
# print("Testing accuracy :",accuracy_score(y_test,y_pred_test))


