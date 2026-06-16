##  IMPORTING ONLINE DATASETS
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.metrics import accuracy_score

newsgroups= fetch_20newsgroups(subset='all',shuffle=True,random_state=42)
# print(newsgroups.DESCR)

# print(newsgroups.target_names) ##  LISTS OF LABELS

categories=['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
# print(len(categories))           ## 20

news_train=fetch_20newsgroups(subset='train',categories=categories)
news_test=fetch_20newsgroups(subset='test',categories=categories)
# print(news_train)
# print("Total Data :",len(newsgroups.data))        ## 18846
# print("Training Data :",len(news_train.data))     ## 11314
# print("Testing Data :",len(news_test.data))       ## 7532

# print(news_train.data[0])
# print(news_train.data[10096])
# print(news_train.data[-1])

## CREATING A MODEL BASED ON MULTINOMIAL NAIVE BAYERS

model= make_pipeline(TfidfVectorizer(),MultinomialNB())

model.fit(news_train.data,news_train.target)
results= model.predict(news_test.data)
# print(results)

# print("Acurracy ; ", accuracy_score(news_test.target,results))
# print(metrics.classification_report(news_test.target,results,target_names=news_test.target_names))

def predict_category(s,train=news_train,model=model):
    pred=model.predict([s])
    return train.target_names[pred[0]]

# print(predict_category('Jesus Christ'))
# print(predict_category('Python is way better then c,c+_,java'))
# print(predict_category('who is narendra modi'))

# print(news_train.data[0])
# print(predict_category(" The doors were really small"))
# print(predict_category("state of hormus"))
# print(predict_category("Royal Challenger Bengluru"))
print(predict_category("Iphone cheaper"))