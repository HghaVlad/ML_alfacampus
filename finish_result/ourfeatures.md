## Наши фичи

- [Создание первого датасета](#создание-первого-датасета)
  - [Подготовка функций](#подготовка-функций)
  - [Получение features](#функция-для-получения-features)
  - [Сохранение датасета](#создание-и-сохранение-датасета)
- [Обновление датасета](#обновление-датасета)
  - [Добавление столбцов](#добавление-столбцов)
  - [Удаление столбцов](#удаление-столбцов)

### Создание первого датасета

```
import pandas as pd
import numpy as np
```

##### Подготовка функций
```
def num_unique_values(x):
    return len(x.unique())

def get_operation_type_group(x):
    my_uniq = x.value_counts()
    return list(({k:my_uniq[k] if k in my_uniq else 0 for k in range(1, 5)} ).values())

def get_operation_kind(x):
    my_uniq = x.value_counts()
    return list(({k:my_uniq[k] if k in my_uniq else 0 for k in range(1, 8)} ).values())

def get_income_flag(x):
    my_uniq = x.value_counts()
    return list(({k:my_uniq[k] if k in my_uniq else 0 for k in range(1, 4)} ).values())

def get_payment_system(x):
    my_uniq = x.value_counts()
    return list(({k:my_uniq[k] if k in my_uniq else 0 for k in range(1, 8)} ).values())

def get_ecommerce_flag(x):
    my_uniq = x.value_counts()
    return list(({k:my_uniq[k] if k in my_uniq else 0 for k in range(1, 4)} ).values())

operation_type_group_columns = ["operation_type_group"+str(x) for x in list(my_xtrain.operation_type_group.value_counts().index)]
operation_kind_columns = ["operation_kind"+str(x) for x in list(my_xtrain.operation_kind.value_counts().index)]
income_flag_columns = ["income_flag"+str(x) for x in list(my_xtrain.income_flag.value_counts().index)]
payment_system_columns = sorted(["payment_system"+str(x) for x in list(my_xtrain.payment_system.value_counts().index)])
ecommerce_flag_columns = ["ecommerce_flag"+str(x) for x in list(my_xtrain.ecommerce_flag.value_counts().index)]
todelete_columns = ["operation_type_group_get_operation_type_group", "operation_kind_get_operation_kind", "income_flag_get_income_flag", "payment_system_get_payment_system", "ecommerce_flag_get_ecommerce_flag"]
```

##### Подготовка запроса
```
agg_features = {
        "transaction_number": "max",
        "amnt": ["min", "max", "mean", "median"],
        "currency": num_unique_values,
        'operation_type_group': [get_operation_type_group, "median"],
        "operation_kind": [get_operation_kind, "median"],
        'income_flag' : [get_income_flag, "median"],
        "payment_system": [get_payment_system, "median"],
        "ecommerce_flag": [get_ecommerce_flag, "mean"],
        'days_before': ['min', 'mean', 'max'],
        "country": [num_unique_values, "median"],
        'city': [num_unique_values, "median"],
        'day_of_week': [num_unique_values, "mean"],
}
```

##### Функция для получения features
```
def get_new_features(dataset, target):
    features = dataset.groupby("app_id", as_index=False).agg(agg_features)
    
    features[operation_type_group_columns] = features["operation_type_group"]["get_operation_type_group"].to_list()
    features[operation_kind_columns] = features["operation_kind"]["get_operation_kind"].to_list()
    features[income_flag_columns] = features["income_flag"]["get_income_flag"].to_list()
    features[payment_system_columns] = features["payment_system"]["get_payment_system"].to_list()
    features[ecommerce_flag_columns] = features["ecommerce_flag"]["get_ecommerce_flag"].to_list()

    features.columns = ['_'.join(col).strip('_') for col in features.columns.values]
    features = features.drop(todelete_columns, axis=1)
    features = features.join(target.set_index("app_id"), "app_id")
    return features.drop(columns=[ "flag"]), features[["app_id", "flag"]]
```

##### Создание и сохранение датасета
```
my_xtrain = pd.read_parquet("dataset_train_small.parquet")
my_ytrain = pd.read_parquet("target_train_small.parquet")

new_trainx, new_trainy = get_new_features(my_xtrain, my_ytrain)

new_trainx.to_csv("our_features\my_main_featureX.csv")
new_trainy.to_csv("our_features\my_main_featureY.csv")
```

### Обновление датасета

### Добавление столбцов

##### Функция для создания столбцов
```
def new_columns(new_agg: dict, origin_dataset: pd.DataFrame):
    new_dataset = origin_dataset.groupby("app_id").agg(new_agg)
    return new_dataset
```

##### Получение количества уникальных значений
```
    dataset_trainx['weekofyear'].value_counts()
```

##### Подготовка функций для обработки
```
def get_week_of_year_k(x):
    my_uniq = x.value_counts()
    return list(({k:my_uniq[k] if k in my_uniq else 0 for k in range(1, 28)} ).values())
```

##### Вызов функции и обновление столбцов
```
new_columns_tofeaturex = new_columns({ "weekofyear": get_week_of_year_k, "days_before": {"mean", "count"} }, dataset_trainx)

mynew_columns = sorted(["my_weekofyear_"+str(x) for x in list(dataset_testx.weekofyear.value_counts().index)])
new_columns_tofeaturex[mynew_columns] = new_columns_tofeaturex["weekofyear"].to_list()
```

##### Объединение с предыдущей версией датасета и сохранения новой
```
new_columns_tofeaturex.columns = ['_'.join(col).strip('_') for col in new_columns_tofeaturex.columns.values]
new_featurex = my_trainx_feature.merge(new_columns_tofeaturex, on="app_id")

new_featurex.to_csv("ourfeatures/my_new_versionof_featureX.csv", index=False)
```

### Удаление столбцов

##### Функция для удаления столбцов
```
def delete_columns(column_list: list, dataset: pd.DataFrame):
    new_dataset = dataset.drop(column_list, axis=1)
    return new_dataset
```

##### Подготовка столбцов и получения обновленного датасета
```
columns_to_delete = ["income_flag_median", "day_before_std"]
new_featurex = delete_columns(columns_to_delete, my_trainx_feature)

new_featurex.to_csv("ourfeatures/my_new_versionof_featureX.csv", index=False)
```