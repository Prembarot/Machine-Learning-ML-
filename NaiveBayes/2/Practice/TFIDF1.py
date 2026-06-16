from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

stmt1 = "data science machine"
stmt2 = "machine learning deep learning"
stmt3 = "student name list roll number"

mydf=pd.DataFrame({'First_Stmt' : [stmt1],'Second_Stmt' : [stmt2], 'Third_Stmt' : [stmt3]})
print(mydf.iloc[0])
tdidf_vectorizer= TfidfVectorizer()
doc_vec= tdidf_vectorizer.fit_transform(mydf.iloc[0])
# print(doc_vec)
# print(doc_vec.toarray())
# print(doc_vec.toarray().transpose())

mydf1=pd.DataFrame(doc_vec.toarray().transpose(),index=tdidf_vectorizer.get_feature_names_out())
mydf1.columns=mydf.columns
print(mydf1)