## Наши модели

- [RandomForestClassifier](#модель-randomforestclassifier)
- [CatBoostRegressor](#модель-catboostregressor)
- [CatBoostClassifier](#модель-catboostclassifier)
- [Нормирование данных](#использование-модели-с-нормированием-данных)


### Модель RandomForestClassifier()

Создание и обучение модели
```
x_train, x_test, y_train, y_test =train_test_split(my_xtrain, my_ytrain, train_size=0.1)
model = RandomForestClassifier(n_estimators=200,
                               bootstrap = True,
                               max_features = 'sqrt' ,max_depth=10,min_samples_split=5, random_state=1,criterion="log_loss" )
model.fit(x_train, y_train["flag"])
```

Тестирование модели
```
pred = model.predict(x_test)
rmse = (np.sqrt(mean_squared_error(y_test['flag'], pred)))
r2 = r2_score(y_test['flag'], pred)
score = model.score(x_test, y_test['flag'])
local_score = model.score(x_train, y_train['flag'])

print("Testing performance")
print("RMSE: {:.2f}".format(rmse))
print("R2: {:.2f}".format(r2))
print("Score: {:.4f}".format(score))
print("Local Score: {:.4f}".format(local_score))
print("Best params: ", model.get_params())
```

Запись результата
```
y_test_pred = model.predict_proba(contest_x)[:, 1]
contest_y["flag"] = y_test_pred
contest_y.to_csv("results/my_result.csv", index=False)
```

### Модель CatBoostRegressor
Создание и обучение модели
```
x_train, x_test, y_train, y_test = train_test_split(X_scaled, my_ytrain, test_size=0.1)
dataset = cb.Pool(x_train, y_train['flag'])
model = cb.CatBoostRegressor(loss_function="Poisson")
grid = {'iterations': [194, 193],
        'learning_rate': [ 0.15],
        'depth': [10, 9],
        'l2_leaf_reg': [3, 4 ]}
model.grid_search(grid, dataset)
```

Тестирование модели
```
pred = model1.predict(x_test)
rmse = (np.sqrt(mean_squared_error(y_test['flag'], pred)))
r2 = r2_score(y_test['flag'], pred)
score = model1.score(x_test, y_test['flag'])
local_score = model1.score(x_train, y_train['flag'])
aucs = roc_auc_score(y_test['flag'], pred)

print("Testing performance")
print("RMSE: {:.2f}".format(rmse))
print("R2: {:.2f}".format(r2))
print("Score: {:.4f}".format(score))
print("Local Score: {:.4f}".format(local_score))

print("Best params: ", model1._get_params())
```

Запись результата
```
prediction = model.predict(contest_x)
contest_y["flag"] = prediction
contest_y.to_csv("results/my_result.csv", index=False)
```

### Модель CatBoostClassifier

Создание и обучение модели
```
x_train, x_test, y_train, y_test = train_test_split(X_scaled, my_ytrain, test_size=0.1)
dataset = cb.Pool(x_train, y_train['flag'])
model2 = cb.CatBoostClassifier()
grid = {'iterations': [190, 180],
        'learning_rate': [0.03, 0.1],
        'depth': [4,  6,],
        'l2_leaf_reg': [0.2, 1, 3]}
model2.grid_search(grid, dataset)
```

Тестирование модели
```
pred = model2.predict(x_test)
rmse = (np.sqrt(mean_squared_error(y_test['flag'], pred)))
r2 = r2_score(y_test['flag'], pred)
score = model2.score(x_test, y_test['flag'])
local_score = model2.score(x_train, y_train['flag'])
print("Testing performance")
print("RMSE: {:.2f}".format(rmse))
print("R2: {:.2f}".format(r2))
print("Score: {:.4f}".format(score))
print("Local Score: {:.4f}".format(local_score))
print("Best params: ", model2._get_params())
```

Запись результата
```
prediction = model2.predict_proba(contest_x)[:,1]
contest_y["flag"] = prediction
contest_y.to_csv("results/my_result.csv", index=False)
```

### Использование модели с нормированием данных

Подготовка данных к нормализации
```
scaler = StandardScaler()
X_scaled = scaler.fit_transform(my_xtrain)
x_train, x_test, y_train, y_test = train_test_split(X_scaled, my_ytrain, test_size=0.1)
```

Создание модели
```
model2 = cb.CatBoostClassifier()
grid = {'iterations': [190, 180],
        'learning_rate': [0.03, 0.1],
        'depth': [4,  6,],
        'l2_leaf_reg': [0.2, 1, 3]}
model2.grid_search(grid, dataset)
```