import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV

data = load_breast_cancer()
X = data.data
y = data.target

print("Kształt X:", X.shape)
print("Kształt y:", y.shape)

# Podział na train, validation,  test (70% / 15% / 15%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("Zbiór treningowy:  ", X_train.shape)
print("Zbiór walidacyjny: ", X_val.shape)
print("Zbiór testowy:     ", X_test.shape)

# Skalowanie cech
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)  # fit + transform na train
X_val_scaled   = scaler.transform(X_val)         # tylko transform!
X_test_scaled  = scaler.transform(X_test)        # tylko transform!

print("Średnia przed skalowaniem: ", X_train[:, 0].mean().round(2))
print("Średnia po skalowaniu:     ", X_train_scaled[:, 0].mean().round(2))
print("Odch. std po skalowaniu:   ", X_train_scaled[:, 0].std().round(2))


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Baseline = 63%
baseline = max(np.bincount(y_train)) / len(y_train)
print(f"Baseline naiwny: {baseline:.4f} ({baseline*100:.1f}%)")

# Sieć neuronowa
model = MLPClassifier(
    hidden_layer_sizes=(64,),
    activation='relu',
    max_iter=1000,
    random_state=42
)
# inicjacja nauki
model.fit(X_train_scaled, y_train)

# Ocena na zbiorze walidacyjnym
y_pred = model.predict(X_val_scaled)
acc = accuracy_score(y_val, y_pred)
print(f"Dokładność modelu: {acc:.4f} ({acc*100:.1f}%)")
print(f"Poprawa vs baseline: +{(acc - baseline)*100:.1f}%")

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Pełny raport
print("\n\n=== RAPORT KLASYFIKACJI ===")
print(classification_report(y_val, y_pred, target_names=['Malignant', 'Benign']))

# Macierz konfuzji
cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Malignant', 'Benign'])
disp.plot(cmap='Blues')
plt.title('Macierz konfuzji - zbiór walidacyjny')
plt.savefig('results/plots/confusion_matrix.png', bbox_inches='tight')
plt.show()


# Słownik przechowujący wyniki eksperymentów
wyniki = {}

# Lista architektur do przetestowania
architektury = {
    'mała (32,)':         (32,),
    'bazowa (64,)':       (64,),
    'duża (128,)':        (128,),
    'dwie warstwy (64,32)':    (64, 32),
    'trzy warstwy (128,64,32)': (128, 64, 32)
}

for nazwa, architektura in architektury.items():
    m = MLPClassifier(
        hidden_layer_sizes=architektura,
        activation='relu',
        max_iter=1000,
        random_state=42
    )
    m.fit(X_train_scaled, y_train)
    y_pred_val = m.predict(X_val_scaled)
    
    acc = accuracy_score(y_val, y_pred_val)
    rec = recall_score(y_val, y_pred_val, pos_label=0)
    
    wyniki[nazwa] = {'accuracy': acc, 'recall_malignant': rec}
    print(f"{nazwa:30s} → Accuracy: {acc:.4f} | Recall Malignant: {rec:.4f}")


param_grid = {
    'hidden_layer_sizes': [(32,), (64,), (128,), (64, 32)],
    'activation':         ['relu', 'tanh'],
    'alpha':              [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01]
}

grid_search = GridSearchCV(
    MLPClassifier(max_iter=1000, random_state=42),
    param_grid,
    cv=5,
    scoring='recall_macro',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print("\n=== NAJLEPSZE PARAMETRY ===")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_val_scaled)
acc_best = accuracy_score(y_val, y_pred_best)
rec_best = recall_score(y_val, y_pred_best, pos_label=0)

print(f"\nNajlepszy model → Accuracy: {acc_best:.4f} | Recall Malignant: {rec_best:.4f}")

print("\n" + "="*50)
print("=== FINALNA EWALUACJA NA ZBIORZE TESTOWYM ===")
print("="*50)

y_pred_test = best_model.predict(X_test_scaled)

acc_test = accuracy_score(y_test, y_pred_test)
rec_test = recall_score(y_test, y_pred_test, pos_label=0)

print(f"\nAccuracy:          {acc_test:.4f} ({acc_test*100:.1f}%)")
print(f"Recall Malignant:  {rec_test:.4f} ({rec_test*100:.1f}%)")

print("\n=== RAPORT KLASYFIKACJI ===")
print(classification_report(y_test, y_pred_test, target_names=['Malignant', 'Benign']))

cm_test = confusion_matrix(y_test, y_pred_test)
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=['Malignant', 'Benign'])
disp_test.plot(cmap='Blues')
plt.title('Macierz konfuzji — zbiór testowy (finalna ewaluacja)')
plt.savefig('results/plots/confusion_matrix_test.png', bbox_inches='tight')
plt.show()

print("\n=== PORÓWNANIE WALIDACJA vs TEST ===")
print(f"Accuracy  — walidacja: {acc_best:.4f} | test: {acc_test:.4f}")
print(f"Recall    — walidacja: {rec_best:.4f} | test: {rec_test:.4f}")