# Importações necessárias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE
import shap
import joblib

# 1. Carregar os dados localmente
print("Carregando os dados localmente...")
file_path = "data/staging/staging_churn.csv"  # Caminho local do CSV
data = pd.read_csv(file_path)
print("Primeiras linhas do dataset:")
print(data.head())

# 2. Preparar os dados
print("\nPreparando os dados...")
data = data.select_dtypes(include=["number"])  # Selecionar apenas colunas numéricas
X = data.drop(columns=["Exited"])  # Variável target 'Exited'
y = data["Exited"]

# 3. Aplicar SMOTE para balancear as classes
print("\nAplicando SMOTE para balancear as classes...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("Distribuição das classes após SMOTE:")
print(y_resampled.value_counts())

# 4. Dividir os dados em treino e teste
print("\nDividindo os dados em treino e teste...")
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42
)

# 5. Treinar o modelo RandomForest
print("\nTreinando o modelo RandomForest...")
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=42
)
rf_model.fit(X_train, y_train)

# 6. Avaliar o modelo no conjunto de teste
print("\n--- Avaliando o Modelo no Conjunto de Teste ---")
y_pred = rf_model.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

# 7. Matriz de Confusão ajustada
print("\nPlotando Matriz de Confusão...")
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest", cmap="Blues")  # Fundo azul claro
plt.title("Matriz de Confusão no Conjunto de Teste", fontsize=14)
plt.colorbar()

# Colocar rótulos nos eixos
classes = ["Classe 0", "Classe 1"]
tick_marks = [0, 1]
plt.xticks(tick_marks, classes, fontsize=12)
plt.yticks(tick_marks, classes, fontsize=12)
plt.xlabel("Classe Prevista", fontsize=12)
plt.ylabel("Classe Real", fontsize=12)

# Adicionar os valores dentro das células com fonte preta
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            f"{cm[i, j]}",
            ha="center",
            va="center",
            color="black",  # Fonte sempre preta
            fontsize=12,
            fontweight="bold",
        )

plt.tight_layout()
plt.show()

# 8. Importância das Variáveis Ordenada
print("\nPlotando Importância das Variáveis (Ordenada)...")
importances = rf_model.feature_importances_
features = X.columns

# Ordenar as variáveis por importância
sorted_idx = importances.argsort()[::-1]
importances_sorted = importances[sorted_idx]
features_sorted = features[sorted_idx]

fig, ax = plt.subplots(figsize=(10, 8))

# Plotar as barras em azul mais escuro e ordenadas
bars = ax.barh(features_sorted, importances_sorted, color="#4682B4", edgecolor="black", linewidth=1.2)
ax.invert_yaxis()  # Inverter a ordem para maior no topo

# Adicionar os valores numéricos ao final das barras
for bar in bars:
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.2f}", va="center", fontsize=10, color="black")

# Configurar título e rótulos
ax.set_title("Importância das Variáveis no Modelo RandomForest", fontsize=14)
ax.set_xlabel("Importância Normalizada", fontsize=12)
ax.set_ylabel("Variáveis", fontsize=12)

# Ajustar layout
plt.tight_layout()
plt.show()

# 9. Explicação SHAP - Melhorada
print("\nGerando Explicação SHAP...")
explainer = shap.Explainer(rf_model, X_train)
shap_values = explainer(X_test)

# SHAP Summary Plot com maior espessura nos pontos
print("Plotando Gráfico SHAP Ajustado...")
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values, X_test, feature_names=X.columns, plot_type="dot", 
    alpha=0.8, show=True
)

# 10. Salvar o modelo treinado
output_path = "models/random_forest_churn.pkl"
joblib.dump(rf_model, output_path)
print(f"\nModelo RandomForest salvo em: {output_path}")