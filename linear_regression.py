import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("D:\ML codes\linear_reg.csv")

df['X_2'] = df['X'] ** 2
df['Y_2'] = df['Y'] ** 2
df['XY'] = df['X'] * df['Y']
n = len(df)

a = ((sum(df['Y']) * sum(df['X_2'])) - ((sum(df['X'])) * sum(df['XY']))) / ((n * sum(df['X_2'])) - (sum(df['X']) ** 2))
b = ((n * sum(df['XY'])) - (sum(df['X']) * sum(df['Y']))) / ((n * sum(df['X_2'])) - (sum(df['X']) ** 2))

print("Linear Regression equation:")
print("y = ", round(a, 4), "x + ", round(b, 4))


plt.scatter(df["X"], df["Y"])
plt.plot(df['X'], b * df['X'] + a)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression")
plt.show()
x = 15
y = b * x + a
print("Prediction for x =", x, "is:", y)

# import numpy as np 
# import pandas as pd

# data = pd.read_csv('D:\Downloads\hi.csv')
# concepts = np.array(data.iloc[:,0:-1])
# target = np.array(data.iloc[:,-1])  
# def learn(concepts, target): 
#     specific_h = concepts[0].copy()  
#     print("initialization of specific_h \n",specific_h)  
#     general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]     
#     print("initialization of general_h \n", general_h)  

#     for i, h in enumerate(concepts):
#         if target[i] == "yes":
#             print("If instance is Positive ")
#             for x in range(len(specific_h)): 
#                 if h[x]!= specific_h[x]:                    
#                     specific_h[x] ='?'                     
#                     general_h[x][x] ='?'
                   
#         if target[i] == "no":            
#             print("If instance is Negative ")
#             for x in range(len(specific_h)): 
#                 if h[x]!= specific_h[x]:                    
#                     general_h[x][x] = specific_h[x]                
#                 else:                    
#                     general_h[x][x] = '?'        

#         print(" step {}".format(i+1))
#         print(specific_h)         
#         print(general_h)
#         print("\n")
#         print("\n")

#     indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]    
#     for i in indices:   
#         general_h.remove(['?', '?', '?', '?', '?', '?']) 
#     return specific_h, general_h 

# s_final, g_final = learn(concepts, target)

# print("Final Specific_h:", s_final, sep="\n")
# print("Final General_h:", g_final, sep="\n")
