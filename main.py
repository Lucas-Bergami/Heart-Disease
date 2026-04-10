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

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

# ==============================
# 2. CONFIGURAÇÕES
# ==============================
colunas = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

colunas_numericas = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
colunas_categoricas = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

# ==============================
# 3. CRIAR PASTAS
# ==============================
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
base_dir = f"resultados_{timestamp}"

pastas = {
    "histogramas": os.path.join(base_dir, "histogramas"),
    "categoricos": os.path.join(base_dir, "categoricos"),
    "comparacoes": os.path.join(base_dir, "comparacoes"),
    "correlacao": os.path.join(base_dir, "correlacao"),
    "comparacao_modelos": os.path.join(base_dir, "comparacao_modelos"),
}

for pasta in pastas.values():
    os.makedirs(pasta, exist_ok=True)


# ==============================
# 4. FUNÇÃO PARA SALVAR
# ==============================
def salvar_plot(nome, pasta):
    caminho = os.path.join(pasta, f"{nome}.png")
    plt.savefig(caminho, bbox_inches="tight")
    plt.close()


# ==============================
# 5. CARREGAR DATASET (COM FALLBACK)
# ==============================
try:
    print("Tentando carregar via API...")
    heart_disease = fetch_ucirepo(id=45)

    X = heart_disease.data.features
    y = heart_disease.data.targets
    df = pd.concat([X, y], axis=1)

    print("Dataset carregado via API ✔")

except Exception as e:
    print("Erro na API:", e)
    print("Tentando carregar arquivo local...")

    caminho_local = "processed.cleveland.data"

    if not os.path.exists(caminho_local):
        raise FileNotFoundError(
            "Arquivo local não encontrado. Coloque 'processed.cleveland.data' no diretório do script."
        )

    df = pd.read_csv(caminho_local, names=colunas)
    print("Dataset carregado localmente ✔")

# ==============================
# 6. LIMPEZA
# ==============================
df = df.replace("?", np.nan)
df = df.apply(pd.to_numeric)
df = df.fillna(df.median())

# transformar em binário
df["num"] = (df["num"] > 0).astype(int)

# ==============================
# 7. HISTOGRAMAS
# ==============================
df.hist(figsize=(12, 10))
plt.suptitle("Distribuição das Variáveis")
salvar_plot("histograma_geral", pastas["histogramas"])

# ==============================
# DICIONÁRIO DE DESCRIÇÕES
# ==============================
descricoes = {
    "sex": "Sexo (0 = feminino, 1 = masculino). Homens tendem a maior risco cardiovascular.",
    "cp": "Tipo de dor no peito: 1 = angina típica (dor clássica associada ao coração), "
    "2 = angina atípica (dor com algumas características cardíacas), "
    "3 = dor não anginosa (geralmente não relacionada ao coração), "
    "4 = assintomático (sem dor). "
    "Pacientes assintomáticos ou com angina típica tendem a apresentar maior risco de doença coronariana.",
    "exang": "Angina induzida por exercício (0 = não, 1 = sim). Presença indica possível problema cardíaco.",
    "age": "Idade do paciente. Idades maiores aumentam o risco.",
    "trestbps": "Pressão arterial em repouso (mmHg). Valores altos indicam hipertensão.",
    "chol": "Colesterol sérico (mg/dl). Valores altos aumentam risco de aterosclerose.",
    "thalach": "Frequência cardíaca máxima. Valores baixos podem indicar limitação cardíaca.",
    "oldpeak": "Depressão ST. Valores maiores indicam maior chance de isquemia.",
    "ca": "Número de vasos afetados (0–3). Valores maiores indicam maior gravidade.",
    "fbs": "Glicose em jejum >120 mg/dl (0 = não, 1 = sim). Relacionado a diabetes.",
    "restecg": "Resultado do eletrocardiograma (ECG) em repouso: "
    "0 = normal, "
    "1 = alteração nas ondas ST-T (pode indicar isquemia, ou seja, falta de oxigenação no coração), "
    "2 = hipertrofia ventricular esquerda (aumento da espessura do músculo do coração). "
    "Valores 1 e 2 estão associados a maior probabilidade de problemas cardíacos.",
    "slope": "Inclinação do segmento ST durante o exercício: "
    "1 = ascendente (upsloping, geralmente considerado normal), "
    "2 = plano (flat, pode indicar possível problema cardíaco), "
    "3 = descendente (downsloping, frequentemente associado à isquemia, ou seja, falta de oxigenação no coração). "
    "Valores 2 e principalmente 3 estão mais associados à presença de doença coronariana.",
    "thal": "Resultado de exame de perfusão do coração: "
    "3 = normal (fluxo sanguíneo adequado), "
    "6 = defeito fixo (área do coração com dano permanente, geralmente após infarto), "
    "7 = defeito reversível (fluxo sanguíneo reduzido durante esforço, indicando possível obstrução das artérias). "
    "Valores 6 e principalmente 7 estão associados a maior probabilidade de doença coronariana.",
}


# ==============================
# FUNÇÃO PARA ADICIONAR TEXTO
# ==============================
def adicionar_descricao(col):
    if col in descricoes:
        plt.figtext(
            0.5,
            -0.15,
            descricoes[col],
            wrap=True,
            horizontalalignment="center",
            fontsize=9,
        )


# ==============================
# 8. GRÁFICOS CATEGÓRICOS
# ==============================

plt.figure(figsize=(8, 6))
sns.countplot(x="sex", data=df)
plt.title("Distribuição de Sexo")
adicionar_descricao("sex")
salvar_plot("sexo", pastas["categoricos"])

plt.figure(figsize=(8, 6))
sns.countplot(x="cp", data=df)
plt.title("Tipo de Dor no Peito")
adicionar_descricao("cp")
salvar_plot("dor_peito", pastas["categoricos"])

plt.figure(figsize=(8, 6))
sns.countplot(x="exang", hue="num", data=df)
plt.title("Angina vs Doença")
adicionar_descricao("exang")
salvar_plot("angina_vs_doenca", pastas["categoricos"])

# ==============================
# 9. COMPARAÇÕES
# ==============================

# NUMÉRICOS → BOXPLOT
for col in colunas_numericas:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="num", y=col, data=df)
    plt.title(f"{col} vs Doença")
    adicionar_descricao(col)
    salvar_plot(f"{col}_boxplot", pastas["comparacoes"])

# CATEGÓRICOS → COUNTPLOT
for col in colunas_categoricas:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=col, hue="num", data=df)
    plt.title(f"{col} vs Doença")
    adicionar_descricao(col)
    salvar_plot(f"{col}_countplot", pastas["comparacoes"])

plt.figure(figsize=(9, 6))

# calcular média e contagem
agrupado = df.groupby("ca")["num"].agg(["mean", "count"]).reset_index()

# gráfico
sns.barplot(x="ca", y="mean", data=agrupado)

# adicionar valores acima das barras
for i, row in agrupado.iterrows():
    plt.text(
        i,
        row["mean"] + 0.02,
        f"{row['mean']:.2f}\n(n={int(row['count'])})",
        ha="center",
        fontsize=9,
    )

plt.title("Probabilidade de Doença por Número de Vasos Afetados (ca)")
plt.xlabel("Número de vasos afetados")
plt.ylabel("Probabilidade de doença")

plt.ylim(0, 1)

adicionar_descricao("ca")
salvar_plot("ca_probabilidade", pastas["comparacoes"])

# ==============================
# 10. CORRELAÇÃO
# ==============================
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlação")
salvar_plot("correlacao", pastas["correlacao"])

# ==============================
# 11. MACHINE LEARNING
# ==============================
X = df.drop("num", axis=1)
y = df["num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# salvar resultados
resultados_path = os.path.join(base_dir, "resultados.txt")

with open(resultados_path, "w") as f:
    f.write(f"Acurácia: {accuracy_score(y_test, y_pred)}\n\n")
    f.write(classification_report(y_test, y_pred))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred)))

# ==============================
# 12. K-MEANS
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df["cluster"] = clusters

ari = adjusted_rand_score(y, clusters)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="Set2")
plt.scatter(
    pca.transform(kmeans.cluster_centers_)[:, 0],
    pca.transform(kmeans.cluster_centers_)[:, 1],
    c="red",
    marker="X",
    s=200,
)
plt.title("K-Means PCA")
salvar_plot("kmeans_pca", pastas["comparacao_modelos"])

# ==============================
# 13. COMPARAÇÃO CLUSTER VS REAL
# ==============================
cm_clusters = confusion_matrix(y, clusters)

with open(resultados_path, "a") as f:
    f.write("\n\n=== KMEANS ===\n")
    f.write(f"ARI: {ari}\n")
    f.write(str(cm_clusters))

print(f"\nResultados salvos em: {base_dir}")
