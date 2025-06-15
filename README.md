# ML_final_project
Machine Learning-Based Prediction of the Biological Activity of Chemical Compounds


# Прогнозирование биологической активности химических соединений с использованием методов машинного обучения

## Описание проекта

Цель проекта — разработка моделей машинного обучения для предсказания трёх ключевых биологических показателей химических соединений:

- IC₅₀ — полумаксимальная ингибирующая концентрация;
- CC₅₀ — полумаксимальная цитотоксическая концентрация;
- SI (Selectivity Index) — индекс селективности, рассчитываемый как CC₅₀ / IC₅₀.

Проект включает как задачи регрессии, так и классификации, что позволяет оценивать свойства соединений с разных сторон: количественно и категориально.

Методы реализованы с применением инструментов машинного обучения, хемоинформатики и QSAR-моделирования на основе in silico-подходов.

## Структура репозитория

```
├── report/
│   └── Dmitriy_Kozyrev_ML_final_project_report.pdf
│
├── notebooks/
│   ├── ML_final_project_EDA.ipynb
│   ├── ML_final_project_IC50_regression.ipynb
│   ├── ML_final_project_CC50_regression.ipynb
│   ├── ML_final_project_SI_regression.ipynb
│   ├── ML_final_project_IC50_classification.ipynb
│   ├── ML_final_project_CC50_classification.ipynb
│   └── ML_final_project_SI_classification.ipynb
│
├── data/
│   ├── df.csv
│   ├── df_bin.csv
│   ├── df_cut.csv
│   ├── df_cut_bin.csv
│   ├── report_df.txt
│   └── ...
│
├── models/
│   ├── regression_IC50_catboost_final_model.pkl
│   ├── regression_CC50_lightgbm_final_model.pkl
│   ├── regression_SI_catboost_final_model.pkl
│   ├── IC50_classifier_catboost_final_model.pkl
│   └── ...
│
├── results/
│   ├── regression_comparison_metrics_IC50.xlsx
│   ├── classification_comparison_metrics_CC50.xlsx
│   ├── df_correlation_matrix.png
│   ├── zscore_boxplot.png
│   └── ...
└── README.md
```

## Используемые алгоритмы

### Регрессия

- KNeighborsRegressor
- Random Forest
- Gradient Boosting (включая XGBoost, LightGBM, CatBoost)
- HistGradientBoosting, AdaBoost
- Стэкинг с мета-моделью Linear Regression
- Глубокие нейронные сети (MLPRegressor, PyTorch)

### Классификация

- Logistic Regression
- Random Forest, Gradient Boosting, SVM
- XGBoost, LightGBM, CatBoost
- Многослойный персептрон (MLPClassifier)
- Gaussian Naive Bayes
- K-Nearest Neighbors

## Признаки и дескрипторы

- Физико-химические дескрипторы (молекулярная масса, logP и др.)
- Структурные фрагменты (ароматические кольца, гетероатомы, функциональные группы)
- Topological, geometrical и constitutional дескрипторы
- Bit fingerprints (ECFP, MACCS-ключи)
- Интегральные и топологические дескрипторы: QED, TPSA, EState, BertzCT, BalabanJ, Chi, Kappa, BCUT2D, LabuteASA и др.

## Результаты

### Регрессия

| Метрика         | IC₅₀        | CC₅₀        | SI          |
|-----------------|-------------|-------------|-------------|
| MAE (Test)      | 0.4623      | 0.2833      | 0.3494      |
| RMSE (Test)     | 0.5922      | 0.4175      | 0.5280      |
| R² (Test)       | 0.6480      | 0.5971      | 0.5344      |

### Классификация

| Задача классификации       | Accuracy | ROC AUC |
|----------------------------|----------|---------|
| IC₅₀ (по медиане)          | 0.7662   | 0.8600  |
| CC₅₀ (по медиане)          | 0.7761   | 0.8853  |
| SI (по медиане)            | 0.6766   | 0.7048  |
| SI (>8)                    | 0.7512   | 0.7712  |

## Выводы

Разработанные модели машинного обучения показали высокую эффективность в задачах прогнозирования биологических свойств химических соединений. Значения метрик находятся на уровне или выше типичных показателей, встречающихся в современной литературе по QSAR-прогнозированию. Модели могут быть использованы для ускоренного in silico-скрининга потенциальных кандидатов в лекарственные средства.

## Используемые библиотеки и инструменты

- Python 3.x
- Scikit-learn
- XGBoost, LightGBM, CatBoost
- PyTorch
- RDKit
- Pandas, NumPy, Matplotlib, Seaborn

## Автор

Козырев Дмитрий Анатольевич  
НИЯУ МИФИ, Институт интеллектуальных кибернетических систем  
Кафедра 42, 2025 г.

## Лицензия

Данный проект выполнен в рамках учебной курсовой работы и не предназначен для использования в медицинской практике без соответствующей валидации.
