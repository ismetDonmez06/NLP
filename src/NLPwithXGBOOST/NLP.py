from textblob import TextBlob
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers


from warnings import filterwarnings
filterwarnings('ignore')


import pandas as pd
data = pd.read_csv("train.tsv",sep = "\t")

data["Sentiment"].replace(0, value = "negatif", inplace = True)
data["Sentiment"].replace(1, value = "negatif", inplace = True)

data["Sentiment"].replace(3, value = "pozitif", inplace = True)
data["Sentiment"].replace(4, value = "pozitif", inplace = True)
data = data[(data.Sentiment == "negatif") | (data.Sentiment == "pozitif")]


data.groupby("Sentiment").count()
df = pd.DataFrame()

df["text"] = data["Phrase"]
df["label"] = data["Sentiment"]


##Metin Ön İşleme

#buyuk-kucuk donusumu
df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
#noktalama işaretleri
df['text'] = df['text'].str.replace('[^\w\s]','')
#sayılar
df['text'] = df['text'].str.replace('\d','')
#stopwords
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
sw = stopwords.words('english')
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
#seyreklerin silinmesi
sil = pd.Series(' '.join(df['text']).split()).value_counts()[-1000:]
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sil))
#lemmi
from textblob import Word
#nltk.download('wordnet')
df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))




#Test-Train

train_x, test_x, train_y, test_y = model_selection.train_test_split(df["text"],
                                                                   df["label"],

                                                                    random_state = 1)



encoder = preprocessing.LabelEncoder()

train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)


#Count Vectors
vectorizer = CountVectorizer()
vectorizer.fit(train_x)
x_train_count = vectorizer.transform(train_x)
x_test_count = vectorizer.transform(test_x)
x_train_count.toarray()


xgb = xgboost.XGBClassifier()
xgb_model = xgb.fit(x_train_count,train_y)
accuracy = model_selection.cross_val_score(xgb_model,
                                           x_test_count,
                                           test_y,
                                           cv = 10).mean()



print(accuracy)



