import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("read.csv")

x = df["review"]
y = df["sentiment"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.48, random_state= 80)

vectorizer = TfidfVectorizer()
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

model = LogisticRegression()
model.fit(x_train_vec, y_train)

predicitons = model.predict(x_test_vec)
accuracy = accuracy_score(y_test, predicitons)
print("Accuracy:", accuracy)

def predict_sentiment(review):
    review_vec = vectorizer.transform([review])
    prediciton = model.predict(review_vec)
    return prediciton[0]

print(predict_sentiment("I Loved this movie"))
print(predict_sentiment("Wasted money"))