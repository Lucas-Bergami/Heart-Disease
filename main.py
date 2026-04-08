# ==============================
# 1. IMPORTS
# ==============================
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==============================
# 2. CRIAR ESTRUTURA DE PASTAS
# ==============================

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
base_dir = f"resultados_{timestamp}"

pastas = {
    "histogramas": os.path.join(base_dir, "histogramas"),
    "categoricos": os.path.join(base_dir, "categoricos"),
    "comparacoes": os.path.join(base_dir, "comparacoes"),
    "correlacao": os.path.join(base_dir, "correlacao"),
}

for pasta in pastas.values():
    os.makedirs(pasta, exist_ok=True)

# ==============================
# 3. CARREGAR DATASET
# ==============================
heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features
y = heart_disease.data.targets

df = pd.concat([X, y], axis=1)

# ==============================
# 4. LIMPEZA DOS DADOS
# ==============================
df = df.replace("?", np.nan)
df = df.apply(pd.to_numeric)
df = df.fillna(df.median())

df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)


# ==============================
# 5. FUNÇÃO PARA SALVAR GRÁFICOS
# ==============================
def salvar_plot(nome, pasta):
    caminho = os.path.join(pasta, f"{nome}.png")
    plt.savefig(caminho, bbox_inches="tight")
    plt.close()


# ==============================
# 6. HISTOGRAMAS
# ==============================
df.hist(figsize=(12, 10))
plt.suptitle("Distribuição das Variáveis")
salvar_plot("histograma_geral", pastas["histogramas"])

# ==============================
# 7. GRÁFICOS CATEGÓRICOS
# ==============================

sns.countplot(x="sex", data=df)
plt.title("Distribuição de Sexo")
salvar_plot("sexo", pastas["categoricos"])

sns.countplot(x="cp", data=df)
plt.title("Tipo de Dor no Peito")
salvar_plot("dor_peito", pastas["categoricos"])

sns.countplot(x="exang", hue="num", data=df)
plt.title("Angina vs Doença")
salvar_plot("angina_vs_doenca", pastas["categoricos"])


# ==============================
# COMPARAÇÃO AUTOMÁTICA COM 'num'
# ==============================
# criar função de salvar gráfico se não existir
def salvar_plot_auto(nome, pasta):
    if not os.path.exists(pasta):
        os.makedirs(pasta)
    plt.savefig(os.path.join(pasta, f"{nome}.png"))
    plt.close()


# percorrer todas as colunas exceto 'num'
for col in df.columns:
    if col == "num":
        continue

    plt.figure(figsize=(8, 6))

    if pd.api.types.is_numeric_dtype(df[col]):
        sns.boxplot(x="num", y=col, data=df)
        plt.title(f"{col} vs Doença (Boxplot)")
        salvar_plot_auto(f"{col}_vs_doenca_boxplot", pastas["comparacoes"])

        sns.violinplot(x="num", y=col, data=df)
        plt.title(f"{col} vs Doença (Violin)")
        salvar_plot_auto(f"{col}_vs_doenca_violin", pastas["comparacoes"])

    # se categórica -> countplot
    else:
        sns.countplot(x=col, hue="num", data=df)
        plt.title(f"{col} vs Doença (Countplot)")
        salvar_plot_auto(f"{col}_vs_doenca_countplot", pastas["comparacoes"])
# ==============================
# EXEMPLO DE REGISTRO
# ==============================

# linha = df.sample(1).iloc[0]

# print("\n=== REGISTRO DO DATASET ===")
# for coluna, valor in linha.items():
#     print(f"{coluna}: {valor}")

# print(heart_disease.variables)

# ==============================
# 9. CORRELAÇÃO
# ==============================
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlação entre Variáveis")
salvar_plot("matriz_correlacao", pastas["correlacao"])

# ==============================
# 10. MACHINE LEARNING
# ==============================

X = df.drop("num", axis=1)
y = df["num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ==============================
# 11. SALVAR RESULTADOS
# ==============================

resultados_path = os.path.join(base_dir, "resultados.txt")

with open(resultados_path, "w") as f:
    f.write("=== RESULTADOS ===\n")
    f.write(f"Acurácia: {accuracy_score(y_test, y_pred)}\n\n")
    f.write("Relatório:\n")
    f.write(classification_report(y_test, y_pred))
    f.write("\nMatriz de Confusão:\n")
    f.write(str(confusion_matrix(y_test, y_pred)))

print(f"\nResultados salvos em: {base_dir}")
