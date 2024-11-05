import numpy as np
import matplotlib.pyplot as plt

df = [[2, 2],
      [3, 2],
      [1, 1],
      [3, 1],
      [1.5, 0.5],
      [1, 0.5],
      [1, 1.5],
      [2.5, 1.5],
      [3, 3],
      [4, 2]]

k = 2
a = df[3]
b = df[7]

c1 = [1]
c2 = [2]
prev1 = []
prev2 = []

while prev1 != c1 and prev2 != c2:
    prev1 = c1
    prev2 = c2
    c1 = []
    c2 = []

    for i in range(len(df)):
        if np.linalg.norm(np.array(a) - np.array(df[i])) < np.linalg.norm(np.array(b) - np.array(df[i])):
            c1.append(df[i])
        else:
            c2.append(df[i])

    a0 = 0
    a1 = 0
    b0 = 0
    b1 = 0

    for i in range(len(c1)):
        a0 += c1[i][0]
        b0 += c1[i][1]

    for i in range(len(c2)):
        a1 += c2[i][0]
        b1 += c2[i][1]

    a0 /= len(c1)
    b0 /= len(c1)
    a1 /= len(c2)
    b1 /= len(c2)
    a = [a0, b0]
    b = [a1, b1]
    
x1, y1 = zip(*c1)
x2, y2 = zip(*c2)

print("Cluster 1:", c1)
print("Cluster 2:", c2)
plt.scatter(x1, y1, color = "red")
plt.scatter(x2, y2, color = "green")
plt.title("Scatter plot between x and y")
plt.xlabel("X - axis")
plt.ylabel("Y - label")
plt.show()        

