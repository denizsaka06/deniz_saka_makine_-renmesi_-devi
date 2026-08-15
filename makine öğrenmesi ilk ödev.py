import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(42)

n = 150

data = {
    "yas": np.random.randint(18, 65, n),
    "gelir": np.random.randint(15000, 80000, n),
    "abonelik_suresi": np.random.randint(1, 60, n),
    "destek_talebi_sayisi": np.random.randint(0, 10, n),
    "sehir": np.random.choice(
        ["Istanbul", "Ankara", "Izmir", "Bursa", "Antalya"], n
    ),
    "uyelik_tipi": np.random.choice(
        ["Standart", "Premium", "Gold"], n
    )
}

data["churn"] = np.random.choice(
    [0, 1],
    n,
    p=[0.65, 0.35]
)

df = pd.DataFrame(data)

print(df.head())
print("\n--- VERİ SETİ İNCELEME ---")

print("\nİlk 5 satır:")
print(df.head())

print("\nSatır ve sütun sayısı:")
print(df.shape)

print("\nSütun bilgileri:")
print(df.info())

print("\nChurn dağılımı:")
print(df["churn"].value_counts())

print("\nChurn yüzdelik dağılımı:")
print(df["churn"].value_counts(normalize=True))
print("\n--- EKSİK DEĞER KONTROLÜ ---")

print(df.isnull().sum())
print("\n--- ÖZNİTELİK ÜRETME ---")

df["gelir_grubu"] = pd.cut(
    df["gelir"],
    bins=[0, 30000, 60000, np.inf],
    labels=["Dusuk", "Orta", "Yuksek"]
)
df["destek_talebi_var_mi"] = (
    df["destek_talebi_sayisi"] > 0
).astype(int)
print("\nYeni oluşturulan öznitelikler:")
print(
    df[
        [
            "gelir",
            "gelir_grubu",
            "destek_talebi_sayisi",
            "destek_talebi_var_mi"
        ]
    ].head()
)
X = df.drop("churn", axis=1)

y = df["churn"]

print("\nX boyutu:", X.shape)
print("y boyutu:", y.shape)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val
)
print("\n--- VERİ BÖLÜMLERİ ---")

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)
numeric_features = [
    "yas",
    "gelir",
    "abonelik_suresi",
    "destek_talebi_sayisi",
    "destek_talebi_var_mi"
]

categorical_features = [
    "sehir",
    "uyelik_tipi",
    "gelir_grubu"
]
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)
logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ]
)
logistic_model.fit(X_train, y_train)
logistic_val_pred = logistic_model.predict(X_val)
logistic_accuracy = accuracy_score(
    y_val,
    logistic_val_pred
)

print("\n--- LOGISTIC REGRESSION ---")
print("Validation Accuracy:", logistic_accuracy)
knn_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]
)
knn_model.fit(X_train, y_train)
knn_val_pred = knn_model.predict(X_val)
knn_accuracy = accuracy_score(
    y_val,
    knn_val_pred
)

print("\n--- KNN ---")
print("Validation Accuracy:", knn_accuracy)
print("\n--- MODEL KARŞILAŞTIRMASI ---")

print("Logistic Regression:", logistic_accuracy)
print("KNN:", knn_accuracy)

if logistic_accuracy >= knn_accuracy:
    best_model = logistic_model
    best_model_name = "Logistic Regression"
else:
    best_model = knn_model
    best_model_name = "KNN"

print("\nValidation sonucuna göre seçilen model:", best_model_name)
test_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, test_pred)

precision = precision_score(y_test, test_pred)

recall = recall_score(y_test, test_pred)

f1 = f1_score(y_test, test_pred)
print("\n--- TEST SONUÇLARI ---")

print("Seçilen Model:", best_model_name)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
cm = confusion_matrix(y_test, test_pred)

print("\n--- CONFUSION MATRIX ---")
print(cm)
plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix")

plt.show()
print("\n--- SONUÇ YORUMU ---")

print(
    f"Validation sonuçlarına göre {best_model_name} modeli "
    "daha başarılı olmuştur."
)

print(
    f"Logistic Regression validation accuracy değeri "
    f"{logistic_accuracy:.2f}, KNN validation accuracy değeri "
    f"{knn_accuracy:.2f} olarak bulunmuştur."
)

print(
    f"Seçilen modelin test accuracy değeri {accuracy:.2f} olmuştur."
)

print(
    "Ancak test sonuçlarında precision, recall ve F1-score değerlerinin "
    "düşük olması, modelin churn eden müşterileri yeterince iyi "
    "tespit edemediğini göstermektedir. Bunun nedeni veri setindeki "
    "churn değerlerinin sınırlı olması ve veri setinin basit şekilde "
    "oluşturulmuş olması olabilir."
)
