import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from configparser import ConfigParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import joblib
import numpy as np

# 1. 加載分類模型和向量化器
print("加載分類模型和向量化器...")
try:
    model = load_model('keras_model.h5')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    classes = joblib.load('label_classes.pkl')
except Exception as e:
    print(f"加載文件時出錯: {e}")
    exit(1)

label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(classes)

# 2. 初始化 Gemini API
print("初始化 Gemini API...")
config = ConfigParser()
config.read("config.ini")
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    google_api_key=config["Gemini"]["API_KEY"],
    convert_system_message_to_human=True,
)

# 3. 定義分類函數
def classify_news(title, content):
    text = f"{title} {content}".strip()
    if not text:
        return "無效的輸入"
    text_tfidf = vectorizer.transform([text]).toarray()
    prediction = model.predict(text_tfidf)
    predicted_class = label_encoder.inverse_transform([prediction.argmax()])
    return predicted_class[0]

# 4. 定義摘要生成函數
def summarize_news(content):
    if not content.strip():
        return "無效的新聞內容，無法生成摘要。"
    response = llm.predict_messages(
        messages=[
            SystemMessage(content="你是專業的摘要生成器，請為以下新聞內容生成摘要："),
            HumanMessage(content=(
                f"請幫我總結以下新聞內文，要求如下：\n"
                f"1. 摘要需限制在100字以內。\n"
                f"2. 請使用繁體中文回答。\n"
                f"新聞內容：\n{content}"
            )),
        ]
    )
    if any(word in response.content for word in ["自殺", "輕生"]):
        response.content += ("\n\n生命誠可貴，輕生不能解決問題。請撥打1925自殺防治安心專線，"
                             "輕生防治諮詢安心專線：1925(依舊愛我)，張老師專線：1980。")
    return response.content

# 5. 接受使用者輸入
news_title = input("新聞標題: ").strip()
news_content = input("新聞內文: ").strip()

# 6. 執行分類與摘要
predicted_category = classify_news(news_title, news_content)
summary = summarize_news(news_content)

# 7. 輸出結果
print("\n=== 分類與摘要結果 ===")
print(f"新聞分類: {predicted_category}")
print(f"新聞摘要: {summary}")
