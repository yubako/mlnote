# Scikit Learn Models

* 決定木モデル

決定木モデルは特徴量の重要度を得られる
```
rf.feature_importances_
```

棒グラフ表示
```
import matplotlib.pyplot as plt
n_features = len(rf.feature_names_in_)
plt.barh(range(n_features), rf.feature_importances_)
plt.yticks(np.arange(n_features), rf.feature_names_in_)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
```

* ランダムフォレスト: RandomForest

[https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

少しずつ異なる決定木を多数集めたアンサンブルモデル

```
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_e)

```

* 勾配ブースティング回帰木: GradientBoosting

[https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)

複数の決定木を組み合わせるアンサンブルモデルだが、１つ前の誤りを次の決定木が修正するようにして浅い(1〜5)の
決定木を順番に作る。
ランダムフォレストと比較するとパラメータの影響を受けやすいが、パラメータさえ適切に設定できればこちらのほうが
良い性能が得られる

```
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

param_grid = {"max_depth": (3,),
              "learning_rate": (0.01,),
              "n_estimators": (200, 300, 500, 700)}
gs = GridSearchCV(estimator=GradientBoostingClassifier(),
                  param_grid=param_grid,
                  scoring="accuracy", cv=10, refit=True, n_jobs=-1)
gs.fit(train_x, train_y)
```
