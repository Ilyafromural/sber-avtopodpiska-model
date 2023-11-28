import logging
import dill
import datetime

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_selector


def filter_data(df):
    df = df.copy()
    columns_to_drop = [
        'session_id',        
        'geo_country'       
    ]
    return df.drop(columns_to_drop, axis=1)


def resolution_type(df):
    df = df.copy()
    df['device_screen_resolution'] = df['device_screen_resolution'].apply(
        lambda x: (int(x.split('x')[0]) * int(x.split('x')[1]))
    )
    return df


def main():
    import pandas as pd
    print('Event Action Prediction Pipeline')

    # Загрузка данных
    df = pd.read_csv('./data/df.csv')    
        
    # Выделение таргета в отдельный датафрейм
    X = df.drop('target', axis=1)
    y = df['target']

    # Преобразование числовых признаков
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Преобразование категориальных признаков
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first', min_frequency=0.02, sparse_output=False))
    ])

    # Собираем преобразования признаков в один трансформер
    column_transformer = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
        ('categorical', categorical_transformer, make_column_selector(dtype_include=object))
    ])

    # Препроцессор для обработки данных
    preprocessor = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('resolution_type', FunctionTransformer(resolution_type)),
        ('column_transformer', column_transformer)
    ])

    # Объявление модели для предсказания
    model = MLPClassifier(
        activation='logistic', max_iter=2000, hidden_layer_sizes=(150, 50),
        learning_rate_init=0.002, learning_rate='adaptive'
    )

    # Создаем пайплайн с моделью    
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Проводим кросс-валидацию и печатаем метрику 
    score = cross_val_score(pipe, X, np.ravel(y), cv=4, scoring='roc_auc')
    print(f'model: {type(model).__name__}, roc_auc_mean: {score.mean():.4f}, roc_auc_std: {score.std():.4f}')
    
    # Обучение модели
    pipe.fit(X, np.ravel(y))

    model_filename = f'./model/prediction_model_{datetime.datetime.now().strftime("%Y%m%d%H%M")}.pkl'

    # Запись пайплайна в файл
    with open(model_filename, 'wb') as file:
        dill.dump({
            'model': pipe,
            'metadata': {
                'name': 'Event action prediction model',
                'author': 'Ilya Pachin',
                'version': 1,
                'date': datetime.datetime.now(),
                'type': type(pipe.named_steps["classifier"]).__name__,
                'roc_auc': score.mean()
            }
        }, file)

    logging.info(f'Model is saved as {model_filename}')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
