import numpy as np
import pandas as pd

df = pd.read_csv("D:\\ML codes\\naive_bayes.csv")

p_yes = 0
p_no = 0

def print_everything():
    print("Conditional Probability Tables:") 
    index = 0
    print("         Yes", "No")
    for j in colour_cond_prob_table:
        if index == 0:
            print("Green ", j)
        elif index == 1:
            print("Blue  ", j)
        else:
            print("Yellow", j)
        index += 1

    print("\n")
    index = 0
    print("           Yes", "No")
    for j in tip_cond_prob_table:
        if index == 0:
            print("Rounded ", j)
        elif index == 1:
            print("Pointy  ", j)
        index += 1

    print("\n")    
    index = 0
    print("           Yes", "No")
    for j in brand_cond_prob_table:
        if index == 0:
            print("Camlin  ", j)
        elif index == 1:
            print("Apsara  ", j)
        index += 1

tip_unique = df["Tip"].unique()
colour_unique = df["Colour"].unique()
brand_unique = df["Brand"].unique()
tip_cond_prob_table = np.zeros((len(tip_unique), 2)).tolist()
colour_cond_prob_table = np.zeros((len(colour_unique), 2)).tolist()
brand_cond_prob_table = np.zeros((len(brand_unique), 2)).tolist()

p_yes = df[df["Sold"] == "Yes"].shape[0]
p_no = df[df["Sold"] == "No"].shape[0]
tip_cond_prob_table[0][0] = df[(df["Tip"] == "Rounded") & (df["Sold"] == "Yes")].shape[0]
tip_cond_prob_table[0][1] = df[(df["Tip"] == "Rounded") & (df["Sold"] == "No")].shape[0]
tip_cond_prob_table[1][0] = df[(df["Tip"] == "Pointy") & (df["Sold"] == "Yes")].shape[0]
tip_cond_prob_table[1][1] = df[(df["Tip"] == "Pointy") & (df["Sold"] == "No")].shape[0]

colour_cond_prob_table[0][0] = df[(df["Colour"] == "Green") & (df["Sold"] == "Yes")].shape[0]
colour_cond_prob_table[0][1] = df[(df["Colour"] == "Green") & (df["Sold"] == "No")].shape[0]
colour_cond_prob_table[1][0] = df[(df["Colour"] == "Blue") & (df["Sold"] == "Yes")].shape[0]
colour_cond_prob_table[1][1] = df[(df["Colour"] == "Blue") & (df["Sold"] == "No")].shape[0]
colour_cond_prob_table[2][0] = df[(df["Colour"] == "Yellow") & (df["Sold"] == "Yes")].shape[0]
colour_cond_prob_table[2][1] = df[(df["Colour"] == "Yellow") & (df["Sold"] == "No")].shape[0]

brand_cond_prob_table[0][0] = df[(df["Brand"] == "Camlin") & (df["Sold"] == "Yes")].shape[0] 
brand_cond_prob_table[0][1] = df[(df["Brand"] == "Camlin") & (df["Sold"] == "No")].shape[0] 
brand_cond_prob_table[1][0] = df[(df["Brand"] == "Apsara") & (df["Sold"] == "Yes")].shape[0] 
brand_cond_prob_table[1][1] = df[(df["Brand"] == "Apsara") & (df["Sold"] == "No")].shape[0] 


print_everything()

count_green = df[df["Colour"] == "Green"].shape[0]
count_blue = df[df["Colour"] == "Blue"].shape[0]
count_yellow = df[df["Colour"] == "Yellow"].shape[0]
for i in range(len(colour_cond_prob_table)):
    
    for j in range(len(colour_cond_prob_table[0])):
        if j == 0:
            if i == 0:
                colour_cond_prob_table[i][j] /= count_green
            elif i == 1:
                colour_cond_prob_table[i][j] /= count_blue
            elif i == 2:
                colour_cond_prob_table[i][j] /= count_yellow
        else:
            if i == 0:
                colour_cond_prob_table[i][j] /= count_green
            elif i == 1:
                colour_cond_prob_table[i][j] /= count_blue
            elif i == 2:
                colour_cond_prob_table[i][j] /= count_yellow

count_round = df[df["Tip"] == "Rounded"].shape[0]
count_pointy = df[df["Tip"] == "Pointy"].shape[0]
for i in range(len(tip_cond_prob_table)):
    for j in range(len(tip_cond_prob_table[0])):
        if j == 0:
            if i == 0:
                tip_cond_prob_table[i][j] /= count_round
            elif i == 1:
                tip_cond_prob_table[i][j] /= count_pointy
                
        else:
            if i == 0:
                tip_cond_prob_table[i][j] /= count_round
            elif i == 1:
                tip_cond_prob_table[i][j] /= count_pointy
            
count_camlin = df[df["Brand"] == "Camlin"].shape[0]
count_apsara = df[df["Brand"] == "Apsara"].shape[0]
for i in range(len(brand_cond_prob_table)):
    for j in range(len(brand_cond_prob_table[0])):
        if j == 0:
            if i == 0:
                brand_cond_prob_table[i][j] /= count_camlin
            elif i == 1:
                brand_cond_prob_table[i][j] /= count_apsara
        else:
            if i == 0:
                brand_cond_prob_table[i][j] /= count_camlin
            elif i == 1:
                brand_cond_prob_table[i][j] /= count_apsara

print("\n")
print_everything()
    
# more probable outcome if tip = pointy, colour = green, brand = apsara
tip = "pointy"
colour = "green"
brand = "apsara"
p_yes_given = tip_cond_prob_table[0][0] * colour_cond_prob_table[0][0] * brand_cond_prob_table[0][0] * (p_yes / 9)
p_no_given = tip_cond_prob_table[0][1] * colour_cond_prob_table[0][1] * brand_cond_prob_table[0][1] * (p_no / 9)

p_yes_given = (p_yes_given) / (p_yes_given + p_no_given)
p_no_given = (p_no_given) / (p_no_given + p_yes_given)

print("\nThe possibility of yes:", p_yes_given)
print("The possibility of no:", p_no_given)

if p_yes_given > p_no_given:
    print("\nPencil which was", tip, colour, brand,"was sold.")
else:
    print("\nPencil which was", tip, colour, brand,"wasn't sold.")
    
    
    
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

X = df[['Tip', 'Colour', 'Brand']]
y = df['Sold']

X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

precision = precision_score(y_test, y_pred, pos_label="Yes")
print(f"Precision: {precision * 100:.2f}%")

recall = recall_score(y_test, y_pred, pos_label="Yes")
print(f"Recall: {recall * 100:.2f}%")

f1 = f1_score(y_test, y_pred, pos_label="Yes")
print(f"F1 Score: {f1 * 100:.2f}%")

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
y_test_bin = [1 if sold == 'Yes' else 0 for sold in y_test]
y_pred_bin = [1 if pred == 'Yes' else 0 for pred in y_pred]
roc_auc = roc_auc_score(y_test_bin, y_pred_bin)
print(f"ROC-AUC Score: {roc_auc:.2f}")
