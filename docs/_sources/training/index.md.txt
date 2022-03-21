# Training

## GridSearch

ハイパーパラメータの組み合わせの最適値探索

```
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {"n_estimators": (10, 50, 100, 200)}
gs = GridSearchCV(estimator=RnadomForestEstimator(),
                  param_grid=param_grid,
                  scoring="accuracy", cv=10, refit=True, n_jobs=-1)
gs.fit(x, y)
print(gs.best_score_)
print(gs.cv_results_)
```

### Dictとして指定するハイパーパラメータ名の取得方法

```
rf = RandomForestClassifier()
rf.get_params().keys()
```

pipelineでも同様
```
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(rf)
pipe.get_params()
---
{'memory': None,
 'steps': [('randomforestclassifier', RandomForestClassifier())],
 'verbose': False,
 'randomforestclassifier': RandomForestClassifier(),
 'randomforestclassifier__bootstrap': True,
 'randomforestclassifier__ccp_alpha': 0.0,
 'randomforestclassifier__class_weight': None,
 'randomforestclassifier__criterion': 'gini',
 'randomforestclassifier__max_depth': None,
 'randomforestclassifier__max_features': 'auto',
 'randomforestclassifier__max_leaf_nodes': None,
 'randomforestclassifier__max_samples': None,
 'randomforestclassifier__min_impurity_decrease': 0.0,
 'randomforestclassifier__min_samples_leaf': 1,
 'randomforestclassifier__min_samples_split': 2,
 'randomforestclassifier__min_weight_fraction_leaf': 0.0,
 'randomforestclassifier__n_estimators': 100,
 'randomforestclassifier__n_jobs': None,
 'randomforestclassifier__oob_score': False,
 'randomforestclassifier__random_state': None,
 'randomforestclassifier__verbose': 0,
 'randomforestclassifier__warm_start': False}
```
