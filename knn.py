import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import math

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df = df.rename(columns={
    "sepal length (cm)": "sepal_length",
    "sepal width (cm)": "sepal_width",
    "petal length (cm)": "petal_length",
    "petal width (cm)": "petal_width",
    "target": "class_label"
})
df.head()

# calculating distance from all other rows and get the top three
# point is sepallength = 4.0, sepalwidth = 3.7, petal_length = 1.3, petal_width = 0.3
k = 3
list_dist = []
def dist_calc():
    for i, r in df.iterrows():
        list_dist.append(math.sqrt((4.0 - r.sepal_length) ** 2
        + (3.7 - r.sepal_width) ** 2 + (1.3 - r.petal_length) ** 2
        + (0.3 - r.petal_width) ** 2))

def get_avg(k):
    sum = 0
    for i in range(k):
        sum += df.iloc[i]["class_label"]
    
    return sum / k
        

dist_calc()
df["dist_target"] = list_dist
df = df.sort_values(by = "dist_target")

print("Predicted class is:", get_avg(3))