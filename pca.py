import numpy as np
import matplotlib.pyplot as plt

df = [[4, 11], [8, 4], [13, 5], [7, 14]]

x1_mean = 0
x2_mean = 0
n = len(df)

for i in range(len(df)):
    x1_mean += df[i][0]
    x2_mean += df[i][1]
    
x1_mean /= len(df)
x2_mean /= len(df)

cov = [[0, 0], 
       [0, 0]]

cov00 = 0
cov01 = 0
cov11 = 0

for i in range(len(df)):
    cov00 += (df[i][0] - x1_mean) ** 2
    cov11 += (df[i][1] - x2_mean) ** 2
    cov01 += (df[i][0] - x1_mean) * (df[i][1] - x2_mean)
    
cov00 /= (n - 1)
cov01 /= (n - 1)
cov11 /= (n - 1)

cov[0][0] = cov00
cov[0][1] = cov[1][0] = cov01
cov[1][1] = cov11

eigen_value, eigen_vectors = np.linalg.eig(np.array(cov))
eigen_vector1 = []
eigen_vector1.append(eigen_vectors[0][1])
eigen_vector1.append(eigen_vectors[1][1])
eigen_vector2 = []
eigen_vector2.append(eigen_vectors[0][0])
eigen_vector2.append(eigen_vectors[1][0])

print("Eigen vector1:", eigen_vector1)
print("Eigen vector2:", eigen_vector2)

pc1 = []
pc2 = []

for i in range(len(df)):
    temp = df[i]
    temp[0] -= x1_mean
    temp[1] -= x2_mean
    transpose_eigen_vector1 = np.transpose(np.array(eigen_vector1))
    pc1.append(np.dot(transpose_eigen_vector1, np.array(temp)))
    
for i in range(len(df)):
    temp = df[i]
    temp[0] -= x1_mean
    temp[1] -= x2_mean
    transpose_eigen_vector2 = np.transpose(np.array(eigen_vector2))
    pc2.append(np.dot(transpose_eigen_vector2, np.array(temp)))
    
print("Principal Component 1:", pc1)
print("Principal Component 2:", pc2)

x_coords = [point[0] for point in df]
y_coords = [point[1] for point in df]

plt.scatter(x_coords, y_coords, color = "red")
plt.show()
plt.scatter(pc1, np.zeros_like(pc1), color = "red")
plt.show()