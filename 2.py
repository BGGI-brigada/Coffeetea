import pandas as pd
import numpy as np

# np.random.seed(37) # если будет перемешивание и для отсутствия случайности разбиения, сейчас используется 100% данных
file_path = 'C:/py proj/1/opros.xlsx'
output_path = 'C:/py proj/1/results.xlsx'

weights = { # веса
    'Ваш пол': 1.0,
    'Возраст': 1.2,
    'Характер': 1.1,
    'Как часто вы берете инициативу в свои руки?': 1.2,
    'Как часто вы пропускаете завтраки?': 1.2,
    'Сколько спите ночью в среднем': 1.5,
    'Гипертония': 2.0
}

num_neighbors = 3 # кол-во соседей, чем больше соседей тем меньше точность получается

data = pd.read_excel(file_path)

columns_to_use = [
    'Ваш пол', 'Возраст', 'Характер', 
    'Как часто вы берете инициативу в свои руки?', 
    'Как часто вы пропускаете завтраки?', 
    'Сколько спите ночью в среднем', 'Гипертония'
]
target_column = 'Что вы предпочитаете?'

data = data[columns_to_use + [target_column]].dropna(subset=[target_column]) # извлекаем data и удаляем строки с отсутств значениями

for column in columns_to_use: # заполняем пропущ. знач. на наиболее част. встречающиеся
    data[column] = data[column].fillna(data[column].mode()[0])

def encodeColumn(column_data): # кодируем в числа
    unique_values = column_data.unique()
    encoding_dict = {val: idx for idx, val in enumerate(unique_values)}
    return column_data.map(encoding_dict), encoding_dict

encoded_data = pd.DataFrame()
encoding_maps = {}

for column in columns_to_use:
    encoded_data[column], encoding_maps[column] = encodeColumn(data[column])

encoded_data[target_column], target_encoding_map = encodeColumn(data[target_column])

X = encoded_data[columns_to_use].values
y = encoded_data[target_column].values

def euclideanDist(x1, x2): # евклидово расст
    weighted_diff = [(x1[i] - x2[i]) ** 2 * weights[columns_to_use[i]] for i in range(len(x1))]
    return np.sqrt(np.sum(weighted_diff))

def kNearestNeighbors(X_train, y_train, X_test, k=num_neighbors): # k-ближ соседи
    predictions = []
    for x_test in X_test:
        distances = [euclideanDist(x_test, x_train) for x_train in X_train]
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[i] for i in k_indices]
        # k_nearest_labels.sort()  # сортировка меток для детерминированности при ничьей, если понадобится
        prediction = max(set(k_nearest_labels), key=k_nearest_labels.count)
        predictions.append(prediction)
    return predictions

predictions = kNearestNeighbors(X, y, X, k=num_neighbors)

decoded_predictions = [list(target_encoding_map.keys())[list(target_encoding_map.values()).index(pred)] for pred in predictions] # декодирование рез-ов

data['Предсказание'] = decoded_predictions
data['Совпадение'] = np.where(data['Предсказание'] == data[target_column], 'Успех', 'Не успех')

def calcAccuracy(y_true, y_pred): # вычисление точности
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

accuracy = calcAccuracy(y, predictions)
print(f"Точность модели: {accuracy * 100:.2f}%")

def classifReport(y_true, y_pred, target_names, output_dict=True): # подробный отчёт
    report = {}
    labels = np.unique(y_true)
    for label in labels:
        true_positive = sum((y_true == label) & (y_pred == label))
        false_positive = sum((y_true != label) & (y_pred == label))
        false_negative = sum((y_true == label) & (y_pred != label))
        support = sum(y_true == label)
        
        precision = true_positive / (true_positive + false_positive + 1e-5)
        recall = true_positive / (true_positive + false_negative + 1e-5)
        f1_score = 2 * precision * recall / (precision + recall + 1e-5)
        
        report[target_names[label]] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1_score,
            'support': support
        }
    if output_dict:
        return report
    else:
        pass

'''
classifReport_dict = classifReport(
    y,
    [target_encoding_map[pred] for pred in data['Предсказание']],
    target_names={v: k for k, v in target_encoding_map.items()},
    output_dict=True
) # подробный отчёт

print("\nПодробный отчёт:")
for label, metrics in classifReport_dict.items():
    if label in target_encoding_map.keys():
        print(f"\nКласс: {label}")
        print(f"Точность: {metrics['precision']:.2f}")
        print(f"Полнота: {metrics['recall']:.2f}")
        print(f"F1-мера: {metrics['f1-score']:.2f}")
        print(f"Поддержка: {metrics['support']}")
'''
data.to_excel(output_path, index=False)

# print(f"Результаты сохранены.")
