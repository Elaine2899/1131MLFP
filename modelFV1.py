import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib
import numpy as np

# 1. 數據讀取與處理
data = pd.read_json('News_Category_Dataset_v3.json', lines=True)
data = data[['headline', 'short_description', 'category']]
data['text'] = data['headline'].fillna('') + " " + data['short_description'].fillna('')

# 查看類別數據分佈
print("類別分佈:")
print(data['category'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['category'], test_size=0.2, random_state=42
)

# 2. 文本向量化
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

# 保存 TF-IDF 向量化器
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# 3. 標籤編碼
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
y_train_onehot = to_categorical(y_train_encoded)
y_test_onehot = to_categorical(y_test_encoded)

# 保存類別列表
classes = np.array(label_encoder.classes_)
joblib.dump(classes, 'label_classes.pkl')

# 4. 建立深度學習模型
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_tfidf.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. 訓練模型
model.fit(X_train_tfidf, y_train_onehot, epochs=10, batch_size=32, validation_split=0.2)

# 6. 評估模型
loss, accuracy = model.evaluate(X_test_tfidf, y_test_onehot)
print(f"測試集準確率: {accuracy}")

# 7. 保存模型
model.save('keras_model.h5')

# 測試新數據
def classify_new_text(text, model, vectorizer, label_encoder):
    text_tfidf = vectorizer.transform([text]).toarray()
    prediction = model.predict(text_tfidf)
    predicted_class = label_encoder.inverse_transform([prediction.argmax()])
    return predicted_class[0]

new_text = "AI technology is rapidly changing the world."
print(f"新文本分類結果: {classify_new_text(new_text, model, vectorizer, label_encoder)}")
