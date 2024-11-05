import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("D:\ML codes\mult_linear_reg.csv")

df["X1_2"] = df["X1"] ** 2
df["X2_2"] = df["X2"] ** 2
df["X1Y"] = df["X1"] * df["Y"]
df["X2Y"] = df["X2"] * df["Y"]
df["X1X2"] = df["X1"] * df["X2"]
n = len(df)

x1 = sum(df["X1"])
x2 = sum(df["X2"])
y = sum(df["Y"])
x1_2 = sum(df["X1_2"]) - ((sum(df["X1"]) ** 2) / n)
x2_2 = sum(df["X2_2"]) - ((sum(df["X2"]) ** 2) / n)
x1y = sum(df["X1Y"]) - ((sum(df["X1"]) * sum(df["Y"])) / n)
x2y = sum(df["X2Y"]) - ((sum(df["X2"]) * sum(df["Y"])) / n)
x1x2 = sum(df["X1X2"]) - ((sum(df["X1"]) * sum(df["X2"])) / n)

b1 = ((x2_2 * (x1y)) - (x1x2 * x2y))
b1 /= ((x1_2 * x2_2) - (x1x2 ** 2))

b2 = (x1_2 * (x2y)) - ((x1x2) * x1y)
b2 /= ((x1_2 * x2_2) - (x1x2 ** 2))

b0 = df["Y"].mean() - (b1 * (df["X1"].mean())) - (b2 * (df["X2"].mean()))

print("Multiple Linear regression equation:")
print("y = ", round(b0, 2), "+ ", round(b1, 2), "* x1 + ", round(b2, 2), " * x2")

x1_in = 4
x2_in = 7

y_ans = b0 + b1 * x1_in + b2 * x2_in
df["Predicted_Y"] = b0 + b1 * df["X1"] + b2 * df["X2"].mean()

fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")
ax.scatter(df["X1"], df["X2"], df["Y"])
X1, X2 = np.meshgrid(df["X1"], df["X2"])
Y = b0 + b1 * X1 + b2 * X2
ax.plot_surface(X1, X2, Y)
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("Y")
ax.set_title("Multiple Linear Regression")
plt.show()

print("Prediction for X1 = 4 and X2 = 7:", round(y_ans, 2))




# model = LinearRegression()
# model.fit(df[["X1", "X2"]], df["Y"])

# print(model.coef_, model.intercept_)

# needed is x2_2 x1_2 x1y x2y
# b0 = y_mean - b1 * x1_mean - b2 * x2_mean
# b1 = (x2_2 * x1y) - ((x1x2) * x2y) /  (x1_2x2_2 - (x1x2)_2)
# b2 = (x1_2 * x2y - x1x2 * x1y) / (x1_2x2_2 - (x1x2)_2)
# x1_2 = X1_2 - X1 square / n
# x2_2 = X2_2 - X2 square/ n
# x1y = X1Y - X1 * Y / n
# x2y = X2Y - X2 * Y / n
# x1x2 = X1X2 - X1 * X2 / n