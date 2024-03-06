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


"""
Değişken Mühendisliği
Count Vectors
TF-IDF Vectors (words, characters, n-grams)
Word Embeddings
TF(t) = (Bir t teriminin bir dökümanda gözlenme frekansı) / (dökümandaki toplam terim sayısı)

IDF(t) = log_e(Toplam döküman sayısı / içinde t terimi olan belge sayısı)

"""


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

#TF-IDF
#wordlevel
tf_idf_word_vectorizer = TfidfVectorizer()
tf_idf_word_vectorizer.fit(train_x)
x_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)
x_train_tf_idf_word.toarray()


# ngram level tf-idf
tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range = (2,3))
tf_idf_ngram_vectorizer.fit(train_x)
x_train_tf_idf_ngram = tf_idf_ngram_vectorizer.transform(train_x)
x_test_tf_idf_ngram = tf_idf_ngram_vectorizer.transform(test_x)

# characters level tf-idf
tf_idf_chars_vectorizer = TfidfVectorizer(analyzer = "char", ngram_range = (2,3))
tf_idf_chars_vectorizer.fit(train_x)
x_train_tf_idf_chars = tf_idf_chars_vectorizer.transform(train_x)
x_test_tf_idf_chars = tf_idf_chars_vectorizer.transform(test_x)



#Makine Öğrenmesi ile Sentiment Sınıflandırması

#Lojistik Regresyon


loj = linear_model.LogisticRegression()
loj_model = loj.fit(x_train_count, train_y)
accuracy = model_selection.cross_val_score(loj_model,
                                           x_test_count,
                                           test_y,
                                           cv = 2).mean()

print("Count Vectors Doğruluk Oranı:", accuracy)
loj = linear_model.LogisticRegression()
loj_model = loj.fit(x_train_count, train_y)
accuracy = model_selection.cross_val_score(loj_model,
                                           x_test_count,
                                           test_y,
                                           cv = 2).mean()

print("Count Vectors Doğruluk Oranı:", accuracy)
loj = linear_model.LogisticRegression()
loj_model = loj.fit(x_train_tf_idf_word,train_y)
accuracy = model_selection.cross_val_score(loj_model,
                                           x_test_tf_idf_word,
                                           test_y,
                                           cv = 2).mean()

print("Word-Level TF-IDF Doğruluk Oranı:", accuracy)
loj = linear_model.LogisticRegression()
loj_model = loj.fit(x_train_tf_idf_ngram,train_y)
accuracy = model_selection.cross_val_score(loj_model,
                                           x_test_tf_idf_ngram,
                                           test_y,
                                           cv = 2).mean()

print("N-GRAM TF-IDF Doğruluk Oranı:", accuracy)
loj = linear_model.LogisticRegression()
loj_model = loj.fit(x_train_tf_idf_chars,train_y)
accuracy = model_selection.cross_val_score(loj_model,
                                           x_test_tf_idf_chars,
                                           test_y,
                                           cv = 2).mean()

print("CHARLEVEL Doğruluk Oranı:", accuracy)
loj = linear_model.LogisticRegression()
loj_model = loj.fit(x_train_tf_idf_chars,train_y)
accuracy = model_selection.cross_val_score(loj_model,
                                           x_test_tf_idf_chars,
                                           test_y,
                                           cv = 2).mean()

print("CHARLEVEL Doğruluk Oranı:", accuracy)

#Naive Bayes
nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(x_train_count,train_y)
accuracy = model_selection.cross_val_score(nb_model,
                                           x_test_count,
                                           test_y,
                                           cv = 2).mean()

print("Count Vectors Doğruluk Oranı:", accuracy)
nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(x_train_tf_idf_word,train_y)
accuracy = model_selection.cross_val_score(nb_model,
                                           x_test_tf_idf_word,
                                           test_y,
                                           cv = 2).mean()

print("Word-Level TF-IDF Doğruluk Oranı:", accuracy)
nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(x_train_tf_idf_ngram,train_y)
accuracy = model_selection.cross_val_score(nb_model,
                                           x_test_tf_idf_ngram,
                                           test_y,
                                           cv = 2).mean()

print("N-GRAM TF-IDF Doğruluk Oranı:", accuracy)
nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(x_train_tf_idf_chars,train_y)
accuracy = model_selection.cross_val_score(nb_model,
                                           x_test_tf_idf_chars,
                                           test_y,
                                           cv = 2).mean()

print("CHARLEVEL Doğruluk Oranı:", accuracy)

#Random Forests
rf = ensemble.RandomForestClassifier()
rf_model = rf.fit(x_train_count,train_y)
accuracy = model_selection.cross_val_score(rf_model,
                                           x_test_count,
                                           test_y,
                                           cv = 2).mean()

print("Count Vectors Doğruluk Oranı:", accuracy)
rf = ensemble.RandomForestClassifier()
rf_model = rf.fit(x_train_tf_idf_word,train_y)
accuracy = model_selection.cross_val_score(rf_model,
                                           x_test_tf_idf_word,
                                           test_y,
                                           cv = 2).mean()

print("Word-Level TF-IDF Doğruluk Oranı:", accuracy)
rf = ensemble.RandomForestClassifier()
rf_model = loj.fit(x_train_tf_idf_ngram,train_y)
accuracy = model_selection.cross_val_score(rf_model,
                                           x_test_tf_idf_ngram,
                                           test_y,
                                           cv = 2).mean()
print("N-GRAM TF-IDF Doğruluk Oranı:", accuracy)
rf = ensemble.RandomForestClassifier()
rf_model = loj.fit(x_train_tf_idf_chars,train_y)
accuracy = model_selection.cross_val_score(rf_model,
                                           x_test_tf_idf_chars,
                                           test_y,
                                           cv = 2).mean()

print("CHARLEVEL Doğruluk Oranı:", accuracy)


#XGBoost
xgb = xgboost.XGBClassifier()
xgb_model = xgb.fit(x_train_count,train_y)
accuracy = model_selection.cross_val_score(xgb_model,
                                           x_test_count,
                                           test_y,
                                           cv = 2).mean()

print("Count Vectors Doğruluk Oranı:", accuracy)
xgb = xgboost.XGBClassifier()
xgb_model = xgb.fit(x_train_tf_idf_word,train_y)
accuracy = model_selection.cross_val_score(xgb_model,
                                           x_test_tf_idf_word,
                                           test_y,
                                           cv = 2).mean()

print("Word-Level TF-IDF Doğruluk Oranı:", accuracy)
xgb = xgboost.XGBClassifier()
xgb_model = xgb.fit(x_train_tf_idf_ngram,train_y)
accuracy = model_selection.cross_val_score(xgb_model,
                                           x_test_tf_idf_ngram,
                                           test_y,
                                           cv = 2).mean()

print("N-GRAM TF-IDF Doğruluk Oranı:", accuracy)
xgb = xgboost.XGBClassifier()
xgb_model = xgb.fit(x_train_tf_idf_chars,train_y)
accuracy = model_selection.cross_val_score(xgb_model,
                                           x_test_tf_idf_chars,
                                           test_y,
                                           cv = 2).mean()

print("CHARLEVEL Doğruluk Oranı:", accuracy)


"""
yeni_yorum = pd.Series("this film is very nice and good i like it")

yeni_yorum = pd.Series("no not good look at that shit very bad")
v = CountVectorizer()
v=v.fit(train_x)
yeni_yorum = v.transform(yeni_yorum)
print(loj_model.predict(yeni_yorum))
"""

