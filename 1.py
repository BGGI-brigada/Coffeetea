# pip install pandas scikit-learn openpyxl
# качаем и радуемся, с библиотекой для сравнения, 40% не оч
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

file_path = 'C:/py proj/1/opros.xlsx'
df = pd.read_excel(file_path)

# признаки для обработки и целевой ответ
features = df[['Ваш пол', 'Возраст', 'Характер', 
               'Как часто вы берете инициативу в свои руки?', 
               'Как часто вы пропускаете завтраки?', 
               'Сколько спите ночью в среднем', 
               'Гипертония', 'Любимое время года?', 
               'Что пьют родители', 'Азартен?']]

target = df['Что вы предпочитаете?']

# заполняем пропуски наиболее часто встречающимися значениями
features.fillna(features.mode().iloc[0], inplace=True)

le = LabelEncoder()
for column in features.columns:
    features[column] = le.fit_transform(features[column])

target = le.fit_transform(target)  # чай/кофе в число

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)  # 20% на тест, 42 - для фиксирования результатов при каждом перезапуске

knn = KNeighborsClassifier(n_neighbors=2)  # количество соседей для анализа
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

report = classification_report(y_test, y_pred, target_names=['Чай', 'Кофе'], output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.rename(columns={
    'precision': 'Точность',    # доля правильных предсказаний для каждого класса
    'recall': 'Полнота',        # способность модели находить все истинные примеры данного класса
    'f1-score': 'F1-мера',      # среднее гармоническое точности и полноты
    'support': 'Поддержка'      # количество истинных примеров каждого класса в данных
}, inplace=True)

df_report.index = df_report.index.str.replace('accuracy', 'Точность модели')  # общая точность предсказаний на всей выборке
df_report.index = df_report.index.str.replace('macro avg', 'Среднее по классам')  # среднее значение метрик между всеми классами
df_report.index = df_report.index.str.replace('weighted avg', 'Взвешенное среднее')  # среднее значение с учетом количества примеров каждого класса

print(df_report)
