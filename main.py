import pandas as pd
import numpy as np


# Нормализация данных вручную
def normalize(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    return (data - min_val) / (max_val - min_val)


# Функция для вычисления Евклидова расстояния
def euclidean_distance(row1, row2):
    return np.sqrt(np.sum((row1 - row2) ** 2))


# Реализация метода KNN
def knn_predict(X_train, y_train, X_test, n_neighbors):
    predictions = []
    for test_row in X_test:
        distances = [euclidean_distance(test_row, train_row) for train_row in X_train]
        nearest_indices = np.argsort(distances)[:n_neighbors]
        nearest_labels = [y_train[i] for i in nearest_indices]
        prediction = max(set(nearest_labels), key=nearest_labels.count)
        predictions.append(prediction)
    return predictions


file_path = 'C:/Users/user/Desktop/Учёба в универе/МАГА/Нейронка/Coffeetea/dataset.xlsx'
df = pd.read_excel(file_path)

# Выбор признаков и целевой переменной
features = df[['Ваш пол', 'Возраст', 'Характер',
               'Как часто вы берете инициативу в свои руки?',
               'Как часто вы пропускаете завтраки?',
               'Сколько спите ночью в среднем',
               'Гипертония', 'Любимое время года?',
               'Что пьют родители', 'Азартен?']]

target = df['Что вы предпочитаете?']

# Заполнение пропусков наиболее часто встречающимися значениями
features.fillna(features.mode().iloc[0], inplace=True)

# Преобразование текстовых данных в числовой формат
label_encoders = {}
for column in features.columns:
    label_encoders[column] = {label: idx for idx, label in enumerate(features[column].astype(str).unique())}
    features[column] = features[column].map(label_encoders[column])
target_mapping = {label: idx for idx, label in enumerate(target.unique())}
target = target.map(target_mapping)

# Нормализация признаков вручную
features = normalize(features.values)

# Разделение данных на обучающую и тестовую выборки
np.random.seed(42)
indices = np.random.permutation(len(features))
split_point = int(0.8 * len(features))
train_indices, test_indices = indices[:split_point], indices[split_point:]

X_train, X_test = features[train_indices], features[test_indices]
y_train, y_test = target.values[train_indices], target.values[test_indices]

# Увеличиваем количество соседей
n_neighbors = 5

# Предсказания модели
y_pred = knn_predict(X_train, y_train, X_test, n_neighbors)

# Оценка точности
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f'Точность модели: {accuracy:.2f}')

# Распределение классов в тестовой выборке
unique, counts = np.unique(y_test, return_counts=True)
print(f'Распределение классов в тестовой выборке: {dict(zip(unique, counts))}')

# Отчет о классификации
classes = ['Чай', 'Кофе']
for class_index, class_name in enumerate(classes):
    true_positive = sum((y_test == class_index) & (y_pred == class_index))
    false_positive = sum((y_test != class_index) & (y_pred == class_index))
    false_negative = sum((y_test == class_index) & (y_pred != class_index))
    support = sum(y_test == class_index)

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f'Класс: {class_name}')
    print(f'  Точность: {precision:.2f}')
    print(f'  Полнота: {recall:.2f}')
    print(f'  F1-мера: {f1_score:.2f}')
    print(f'  Поддержка: {support}')

# Дополнительный вывод предсказаний для проверки
print("\nПредсказания модели:", y_pred)
print("Истинные метки:     ", y_test)
