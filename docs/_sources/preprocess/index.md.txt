# Preprocessing

## クラスラベルのエンコーディング

データ例
```
df = pd.DataFrame([
    ["green", "M", 10.1],
    ["red", "L", 13.5],
    ["blue", "XL", 15.3],
])
df.columns = ["color", "size", "price"]
```

### One-Hot エンコーディング

```
df_encoded = pd.get_dummies(df, columns=["color"])
---
  size  price  color_blue  color_green  color_red
0    M   10.1           0            1          0
1    L   13.5           0            0          1
2   XL   15.3           1            0          0
```

### マッピング

```
size_mapping = {"M": 1, "L": 2, "XL": 3}
df["size"] = df["size"].map(size_mapping)
---
   color  size  price
0  green     1   10.1
1    red     2   13.5
2   blue     3   15.3
```
