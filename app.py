import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from keras.models import load_model

model_path = 'Models/'

# Загрузка датасета
data = pd.read_csv('Data/frauds.csv')
if 'Unnamed: 0' in data.columns:
    data = data.drop(['Unnamed: 0'], axis=1)
X = data.drop(['fraud'], axis=1)
y = data['fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Загрузка моделей
def models():
    model1 = pickle.load(open(model_path + 'knn.pkl', 'rb'))
    model2 = pickle.load(open(model_path + 'kmeans.pkl', 'rb'))
    model3 = pickle.load(open(model_path + 'gradient_boosting_classifier.pkl', 'rb'))
    model4 = pickle.load(open(model_path + 'bagging_classifier.pkl', 'rb'))
    model5 = pickle.load(open(model_path + 'stacking_classifier.pkl', 'rb'))
    model6 = load_model(model_path + 'nr.h5')
    return model1, model2, model3, model4, model5, model6


st.title('Расчётно графичесикая работа ML')
# Навигация
st.sidebar.title('Навигация:')
page = st.sidebar.radio(
    "Выберите страницу",
    ("Разработчик", "Датасет", "Визуализация", "Инференс модели")
)
# Информация о разработчике
def page_developer():
    st.title("Информация о разработчике")
    st.header("Тема РГР:")
    st.write("Разработка Web-приложения для инференса моделей ML и анализа данных")
    col1, col2 = st.columns(2)
    with col1:
        st.header("Фотография")
        st.image("me.jpg", width=150)  # Укажите путь к вашей фотографии
    
    with col2:
        st.header("Контактная информация")
        st.write("ФИО: Ниятшин Эмиль Тимурович")
        st.write("Номер учебной группы: ФИТ-221")
    
    

# Информаиця о нашем датасете
def page_dataset():
    st.title("Информация о наборе данных")

    st.markdown("""
    ## Описание датасета Мошенничество 
    **Файл датасета:** `frauds.csv`

    Цифровые платежи развиваются, но вместе с ними развиваются и киберпреступники.

    Согласно индексу утечек данных, ежедневно похищается более 5 миллионов записей, что является тревожной статистикой, которая показывает, что мошенничество по-прежнему очень распространено как для платежей с использованием карт, так и без них.

    В современном цифровом мире, где ежедневно совершаются триллионы транзакций по картам, выявление мошенничества является сложной задачей.
                
    Данный датасет содержит информацию о мошеннических транзакциях. Содержит следующие столбцы:

    - `distance_from_home`: расстояние от дома, где произошла транзакция.
    - `distance_from_last_transaction`:  расстояние от момента совершения последней транзакции.
    - `ratio_to_median_purchase_price`: отношение покупной цены транзакции к медианной цене покупки.
    - `repeat_retailer`: транзакция, совершенная у того же продавца.
    - `used_chip`: транзакция с помощью чипа (кредитной карты).
    - `used_pin_number`: транзакция, совершенная с использованием PIN-кода.
    - `online_order`: является ли транзакция онлайн-заказом.
    - `fraud`: мошенническая транзакция.           

    **Особенности предобработки данных:**
    - Нормализация числовых данных для улучшения производительности моделей.
    - Масштабирование числовых признаков.
    - Баланс классов
    """)
    
    
# Страница с визуализацией
def page_data_visualization():
    st.title("Визуализации данных")

    st.subheader("Тепловая карта корреляции")
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(14, 11))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=.5, ax=ax_heatmap)
    plt.savefig("heatmap.png")
    st.image("heatmap.png")

    st.subheader("Гистограммы")
    for feature in data:
        fig_hist = plt.figure()
        sns.histplot(data=data, x=feature, kde=True)
        plt.title(f"Гистограмма для {feature}")
        plt.savefig(f"hist_{feature}.png")
        st.image(f"hist_{feature}.png")

    st.subheader("Боксплоты")
    columns = list(data.columns)
    plt.figure(figsize=(12, 10))
    for column in columns:
        plt.subplot(3, 3, columns.index(column) + 1)
        sns.boxplot(data=data, y=column)
        plt.title(column)
    plt.savefig("boxplot.png")
    st.image("boxplot.png")
    
    st.subheader("Матрица диаграмм рассеяния")
    matrix_features = ['fraud', 'distance_from_home','ratio_to_median_purchase_price']
    scatter_matrix_fig = scatter_matrix(data[matrix_features], figsize=(12, 12), alpha=0.8, diagonal='hist')
    plt.savefig("scatter_matrix.png")
    st.image("scatter_matrix.png")


# Страница с инференсом моделей
def page_predictions():
    st.title("Предсказания моделей машинного обучения")

    # Виджет для загрузки файла
    uploaded_file = st.file_uploader("Загрузите ваш CSV файл", type="csv")

    # Интерактивный ввод данных, если файл не загружен
    if uploaded_file is None:
        st.subheader("Введите данные для предсказания:")

        # Интерактивные поля для ввода данных
        input_data = {}
        
        input_data['distance_from_home'] = st.number_input("distance_from_home", min_value=0.0, max_value=100000.0, value=0.577820)
        input_data['distance_from_last_transaction'] = st.number_input("distance_from_last_transaction", min_value=-10000.0, max_value=10000.0, value=-0.139317)
        input_data['ratio_to_median_purchase_price'] = st.number_input("ratio_to_median_purchase_price", min_value=0.0, max_value=100000.0, value=2.34122)
        input_data['repeat_retailer'] = st.number_input("repeat_retailer", min_value=0.0, max_value=1.0, step=1.0, value=1.0)
        input_data['used_chip'] = st.number_input("used_chip", min_value=0.0, max_value=1.0, step=1.0, value=1.0)
        input_data['used_pin_number'] = st.number_input("used_pin_number", min_value=0.0, max_value=1.0, step=1.0, value=0.0)
        input_data['online_order'] = st.number_input("online_order", min_value=0.0, max_value=1.0, step=1.0, value=1.0)
        
        
        if st.button('Сделать предсказание'):
            # Загрузка моделей
            model_ml1, model_ml2, model_ml3, model_ml4, model_ml5, model_ml6 = models()

            input_df = pd.DataFrame([input_data])
            if 'Unnamed: 0' in input_df.columns:
                input_df = input_df.drop(['Unnamed: 0'], axis=1)
            
            
            st.write("Входные данные:", input_df)

            # Используем масштабировщик, обученный на обучающих данных
            scaler = StandardScaler().fit(X_train)
            scaled_input = scaler.transform(input_df)

            # Делаем предсказания
            prediction_ml1 = model_ml1.predict(scaled_input)
            prediction_ml2 = model_ml2.predict(scaled_input)
            prediction_ml3 = model_ml3.predict(scaled_input)
            prediction_ml4 = model_ml4.predict(scaled_input)
            prediction_ml5 = model_ml5.predict(scaled_input)
            prediction_ml6 = (model_ml6.predict(scaled_input) > 0.5).astype(int)

            # Вывод результатов
            st.success(f"Результат предсказания KNN: {prediction_ml1[0]}")
            st.success(f"Результат предсказания K-Means: {prediction_ml2[0]}")
            st.success(f"Результат предсказания GradientBoostingClassifier: {prediction_ml3[0]}")
            st.success(f"Результат предсказания BaggingClassifier: {prediction_ml4[0]}")
            st.success(f"Результат предсказания StackingClassifier: {prediction_ml5[0]}")
            st.success(f"Результат предсказания нейронной сети Tensorflow: {prediction_ml6[0]}")
    else:
        try:
            model_ml1 = pickle.load(open(model_path + 'knn.pkl', 'rb'))
            model_ml2 = pickle.load(open(model_path + 'kmeans.pkl', 'rb'))
            model_ml3 = pickle.load(open(model_path + 'gradient_boosting_classifier.pkl', 'rb'))
            model_ml4 = pickle.load(open(model_path + 'bagging_classifier.pkl', 'rb'))
            model_ml5 = pickle.load(open(model_path + 'stacking_classifier.pkl', 'rb'))
            model_ml6 = load_model(model_path + 'nr.h5')

            # Сделать предсказания на тестовых данных
            predictions_ml1 = model_ml1.predict(X_test)
            predictions_ml2 = model_ml2.predict(X_test)
            predictions_ml3 = model_ml3.predict(X_test)
            predictions_ml4 = model_ml4.predict(X_test)
            predictions_ml5 = model_ml5.predict(X_test)
            predictions_ml6 = model_ml6.predict(X_test).round() # Округление для нейронной сети

            # Оценить результаты
            accuracy_ml1 = accuracy_score(y_test, predictions_ml1)
            accuracy_ml2 = accuracy_score(y_test, predictions_ml2)
            accuracy_ml3 = accuracy_score(y_test, predictions_ml3)
            accuracy_ml4 = accuracy_score(y_test, predictions_ml4)
            accuracy_ml5 = accuracy_score(y_test, predictions_ml5)
            accuracy_ml6 = accuracy_score(y_test, predictions_ml6)

            st.success(f"Точность KNN: {accuracy_ml1}")
            st.success(f"Точность K-Means: {accuracy_ml2}")
            st.success(f"Точность GradientBoostingClassifier: {accuracy_ml3}")
            st.success(f"Точность BaggingClassifier: {accuracy_ml4}")
            st.success(f"Точность StackingClassifier: {accuracy_ml5}")
            st.success(f"Точность нейронной сети Tensorflow: {accuracy_ml6}")
        except Exception as e:
            st.error(f"Произошла ошибка при чтении файла: {e}")



if page == "Разработчик":
    page_developer()
elif page == "Датасет":
    page_dataset()
elif page == "Инференс модели":
    page_predictions()
elif page == "Визуализация":
    page_data_visualization()