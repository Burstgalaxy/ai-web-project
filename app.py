import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from streamlit_drawable_canvas import st_canvas
import cv2

# Настройка страницы
st.set_page_config(page_title="Мой AI Проект", layout="wide")
st.title("🤖 AI Project: Vision & Analytics")

# --- КЭШИРОВАНИЕ МОДЕЛЕЙ ---
# Мы используем кэш, чтобы модели не обучались заново при каждом клике пользователя.
# Обучение произойдет только один раз при запуске сервера.

@st.cache_resource
def load_vision_model():
    # Загрузка MNIST
    (train_img, train_lbl), (test_img, test_lbl) = tf.keras.datasets.mnist.load_data()
    train_img = train_img.reshape((60000, 28, 28, 1)).astype('float32') / 255
    
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Быстрое обучение (можно увеличить эпохи)
    model.fit(train_img, train_lbl, epochs=3, batch_size=64, verbose=0)
    return model

@st.cache_resource
def load_titanic_model():
    titanic = sns.load_dataset('titanic')
    df = titanic[['pclass', 'sex', 'age', 'survived']].dropna()
    le = LabelEncoder()
    df['sex'] = le.fit_transform(df['sex'])
    
    X = df[['pclass', 'sex', 'age']]
    y = df['survived']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Загружаем модели (появится спиннер загрузки при первом старте)
with st.spinner('Загрузка нейросетей... (это займет минуту)'):
    vision_model = load_vision_model()
    titanic_model = load_titanic_model()

# --- ИНТЕРФЕЙС ---

# Вкладки для переключения между проектами
tab1, tab2 = st.tabs(["👁️ Компьютерное зрение (MNIST)", "🚢 Титаник (Аналитика)"])

# === ВКЛАДКА 1: MNIST ===
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("### Нарисуй цифру (0-9)")
        
        # 1. Создаем уникальный ключ для холста, если его нет
        if 'canvas_key' not in st.session_state:
            st.session_state['canvas_key'] = "canvas_v1"

        # 2. Холст (Без стандартной панели инструментов)
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key=st.session_state['canvas_key'], # Ключ привязан к состоянию
            display_toolbar=False,              # <--- ОТКЛЮЧАЕМ СТАРУЮ ПАНЕЛЬ
        )

        # 3. Наша собственная красивая кнопка
        def clear_canvas():
            # Просто меняем ключ, и холст перерисуется заново чистым
            import uuid
            st.session_state['canvas_key'] = str(uuid.uuid4())

        # Кнопка на всю ширину колонки
        st.button("🗑️ ОЧИСТИТЬ", on_click=clear_canvas, type="primary")
        
    with col2:
        st.write("### Результат")
        if canvas_result.image_data is not None:
            img = canvas_result.image_data.astype('uint8')
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if np.max(img_gray) > 0:
                pts = cv2.findNonZero(img_gray)
                x, y, w, h = cv2.boundingRect(pts)
                im_crop = img_gray[y:y+h, x:x+w]
                scale = 20.0 / max(w, h)
                im_resize = cv2.resize(im_crop, (int(w*scale), int(h*scale)))
                new_im = np.zeros((28, 28), np.uint8)
                y_off = (28 - im_resize.shape[0]) // 2
                x_off = (28 - im_resize.shape[1]) // 2
                new_im[y_off:y_off+im_resize.shape[0], x_off:x_off+im_resize.shape[1]] = im_resize
                
                st.image(new_im, caption="Что видит сеть", width=100)
                
                final = new_im.reshape(1, 28, 28, 1).astype('float32') / 255.0
                pred = vision_model.predict(final)
                answ = np.argmax(pred)
                conf = np.max(pred)
                
                st.success(f"Это цифра: **{answ}**")
                st.info(f"Вероятность: {conf:.1%}")
            else:
                st.warning("Холст пуст")

# === ВКЛАДКА 2: ТИТАНИК ===
with tab2:
    st.write("### Прогноз выживания на Титанике")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        sex = st.selectbox("Пол", ["Мужской", "Женский"])
        sex_val = 1 if sex == "Мужской" else 0
        
    with c2:
        age = st.number_input("Возраст", min_value=1, max_value=100, value=25)
        
    with c3:
        pclass_txt = st.selectbox("Класс каюты", ["Эконом (3)", "Средний (2)", "Люкс (1)"])
        if "3" in pclass_txt: pclass = 3
        elif "2" in pclass_txt: pclass = 2
        else: pclass = 1

    if st.button("Узнать шансы"):
        # Формируем данные
        d = pd.DataFrame([[pclass, sex_val, age]], columns=['pclass', 'sex', 'age'])
        
        # Предсказание
        prob = titanic_model.predict_proba(d)[0]
        survival_chance = prob[1]
        
        st.metric(label="Шанс выжить", value=f"{survival_chance:.1%}")
        
        if survival_chance > 0.5:
            st.success("Скорее всего, пассажир ВЫЖИВЕТ")
        else:

            st.error("К сожалению, шансы малы")

