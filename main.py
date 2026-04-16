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
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import wilcoxon, friedmanchisquare

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
    "metricas_modelos": os.path.join(base_dir, "metricas_modelos"),
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
# DISTRIBUIÇÃO DA VARIÁVEL ALVO
# ==============================
plt.figure(figsize=(8, 6))
sns.countplot(x="num", data=df)
plt.title("Distribuição da Variável Alvo (Doença Cardíaca)")
plt.xlabel("Doença (0 = não, 1 = sim)")
plt.ylabel("Quantidade")
plt.figtext(
    0.5,
    -0.15,
    "Mostra quantos pacientes possuem ou não doença cardíaca. "
    "Importante para verificar se o dataset está balanceado.",
    wrap=True,
    ha="center",
)
salvar_plot("distribuicao_alvo", pastas["categoricos"])


# ==============================
# DISTRIBUIÇÃO POR IDADE
# ==============================
plt.figure(figsize=(8, 6))
sns.histplot(df["age"], kde=True)
plt.title("Distribuição de Idade")
plt.xlabel("Idade")
plt.figtext(
    0.5,
    -0.15,
    "Mostra como as idades estão distribuídas no dataset. "
    "Idades mais altas tendem a estar associadas a maior risco cardíaco.",
    wrap=True,
    ha="center",
)
salvar_plot("idade_distribuicao", pastas["histogramas"])


# ==============================
# IDADE vs DOENÇA
# ==============================
plt.figure(figsize=(8, 6))
sns.boxplot(x="num", y="age", data=df)
plt.title("Idade vs Doença")
plt.xlabel("Doença (0 = não, 1 = sim)")
plt.ylabel("Idade")
plt.figtext(
    0.5,
    -0.15,
    "Compara a distribuição de idade entre pacientes com e sem doença. "
    "Pode indicar se pacientes mais velhos têm maior incidência.",
    wrap=True,
    ha="center",
)
salvar_plot("idade_vs_doenca", pastas["comparacoes"])


# ==============================
# SEXO vs DOENÇA (PROPORÇÃO)
# ==============================
plt.figure(figsize=(8, 6))
sns.barplot(x="sex", y="num", data=df)
plt.title("Proporção de Doença por Sexo")
plt.xlabel("Sexo (0 = feminino, 1 = masculino)")
plt.ylabel("Probabilidade de doença")
plt.figtext(
    0.5,
    -0.15,
    "Mostra a proporção de pacientes com doença em cada sexo. "
    "Valores mais altos indicam maior prevalência.",
    wrap=True,
    ha="center",
)
salvar_plot("sexo_proporcao_doenca", pastas["comparacoes"])


# ==============================
# COLESTEROL vs DOENÇA
# ==============================
plt.figure(figsize=(8, 6))
sns.boxplot(x="num", y="chol", data=df)
plt.title("Colesterol vs Doença")
plt.xlabel("Doença (0 = não, 1 = sim)")
plt.ylabel("Colesterol (mg/dl)")
plt.figtext(
    0.5,
    -0.15,
    "Compara níveis de colesterol entre pacientes com e sem doença. "
    "Colesterol alto pode indicar maior risco de problemas cardíacos.",
    wrap=True,
    ha="center",
)
salvar_plot("colesterol_vs_doenca", pastas["comparacoes"])


# ==============================
# PRESSÃO vs DOENÇA
# ==============================
plt.figure(figsize=(8, 6))
sns.boxplot(x="num", y="trestbps", data=df)
plt.title("Pressão Arterial vs Doença")
plt.xlabel("Doença (0 = não, 1 = sim)")
plt.ylabel("Pressão (mmHg)")
plt.figtext(
    0.5,
    -0.15,
    "Analisa se pacientes com doença tendem a ter pressão arterial mais elevada.",
    wrap=True,
    ha="center",
)
salvar_plot("pressao_vs_doenca", pastas["comparacoes"])


# ==============================
# FREQUÊNCIA CARDÍACA vs DOENÇA
# ==============================
plt.figure(figsize=(8, 6))
sns.boxplot(x="num", y="thalach", data=df)
plt.title("Frequência Cardíaca Máxima vs Doença")
plt.xlabel("Doença (0 = não, 1 = sim)")
plt.ylabel("Frequência Cardíaca")
plt.figtext(
    0.5,
    -0.15,
    "Pacientes com menor frequência máxima podem ter limitações cardíacas.",
    wrap=True,
    ha="center",
)
salvar_plot("freq_cardiaca_vs_doenca", pastas["comparacoes"])


# ==============================
# OLDPEAK vs DOENÇA
# ==============================
plt.figure(figsize=(8, 6))
sns.boxplot(x="num", y="oldpeak", data=df)
plt.title("Depressão ST vs Doença")
plt.xlabel("Doença (0 = não, 1 = sim)")
plt.ylabel("Oldpeak")
plt.figtext(
    0.5,
    -0.15,
    "Valores mais altos indicam maior chance de isquemia (falta de oxigênio no coração).",
    wrap=True,
    ha="center",
)
salvar_plot("oldpeak_vs_doenca", pastas["comparacoes"])

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
# 11. MACHINE LEARNING COM K-FOLD
# ==============================

X = df.drop("num", axis=1)
y = df["num"]

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

resultados = {
    "logistic": {"acc": [], "f1": [], "roc": []},
    "tree": {"acc": [], "f1": [], "roc": []},
    "knn": {"acc": [], "f1": [], "roc": []},
    "kmeans": {"ari": []},
}

# MATRIZES DE CONFUSÃO ACUMULADAS
cm_log_total = np.zeros((2, 2))
cm_tree_total = np.zeros((2, 2))
cm_knn_total = np.zeros((2, 2))

for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    print(f"\nFold {fold + 1}")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # ==============================
    # NORMALIZAÇÃO
    # ==============================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==============================
    # LOGISTIC REGRESSION
    # ==============================
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_scaled, y_train)

    y_pred_log = log_model.predict(X_test_scaled)

    resultados["logistic"]["acc"].append(accuracy_score(y_test, y_pred_log))
    resultados["logistic"]["f1"].append(f1_score(y_test, y_pred_log))
    resultados["logistic"]["roc"].append(
        roc_auc_score(y_test, log_model.predict_proba(X_test_scaled)[:, 1])
    )
    cm_log_total += confusion_matrix(y_test, y_pred_log)
    # ==============================
    # DECISION TREE
    # ==============================
    tree_model = DecisionTreeClassifier(random_state=42)
    tree_model.fit(X_train, y_train)

    y_pred_tree = tree_model.predict(X_test)

    resultados["tree"]["acc"].append(accuracy_score(y_test, y_pred_tree))
    resultados["tree"]["f1"].append(f1_score(y_test, y_pred_tree))
    resultados["tree"]["roc"].append(
        roc_auc_score(y_test, tree_model.predict_proba(X_test)[:, 1])
    )
    cm_tree_total += confusion_matrix(y_test, y_pred_tree)
    # ==============================
    # KNN
    # ==============================
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)

    y_pred_knn = knn_model.predict(X_test_scaled)

    resultados["knn"]["acc"].append(accuracy_score(y_test, y_pred_knn))
    resultados["knn"]["f1"].append(f1_score(y_test, y_pred_knn))
    resultados["knn"]["roc"].append(
        roc_auc_score(y_test, knn_model.predict_proba(X_test_scaled)[:, 1])
    )
    cm_knn_total += confusion_matrix(y_test, y_pred_knn)
    # ==============================
    # K-MEANS
    # ==============================
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_train_scaled)
    clusters = kmeans.predict(X_test_scaled)

    ari = adjusted_rand_score(y_test, clusters)
    resultados["kmeans"]["ari"].append(ari)

# ==============================
# 12. RESULTADOS FINAIS
# ==============================
resultados_path = os.path.join(pastas["metricas_modelos"], "resultados.txt")

with open(resultados_path, "w") as f:
    f.write("=== RESULTADOS COM STRATIFIED K-FOLD ===\n\n")

    for modelo in ["logistic", "tree", "knn"]:  # INCLUI KNN
        f.write(f"--- {modelo.upper()} ---\n")
        for metrica in resultados[modelo]:
            media = np.mean(resultados[modelo][metrica])
            std = np.std(resultados[modelo][metrica])
            f.write(f"{metrica}: {media:.4f} ± {std:.4f}\n")
        f.write("\n")

    media_ari = np.mean(resultados["kmeans"]["ari"])
    std_ari = np.std(resultados["kmeans"]["ari"])

    f.write("--- KMEANS ---\n")
    f.write(f"ARI: {media_ari:.4f} ± {std_ari:.4f}\n")

print("\nResultados com K-Fold salvos!")


# ==============================
# COMPARAÇÃO VISUAL DOS MODELOS
# ==============================
# montar tabela
dados_comparacao = []

for modelo in ["logistic", "tree", "knn"]:
    dados_comparacao.append(
        {
            "Modelo": modelo.upper(),
            "Acurácia": np.mean(resultados[modelo]["acc"]),
            "F1-score": np.mean(resultados[modelo]["f1"]),
            "ROC-AUC": np.mean(resultados[modelo]["roc"]),
        }
    )

# KMeans separado (não tem mesmas métricas)
dados_comparacao.append(
    {
        "Modelo": "KMEANS",
        "Acurácia": np.nan,
        "F1-score": np.nan,
        "ROC-AUC": np.nan,
        "ARI": np.mean(resultados["kmeans"]["ari"]),
    }
)

df_comp = pd.DataFrame(dados_comparacao)

# ==============================
# GRÁFICO 1 - MODELOS SUPERVISIONADOS
# ==============================
df_plot = df_comp[df_comp["Modelo"] != "KMEANS"].set_index("Modelo")

plt.figure(figsize=(10, 6))
df_plot.plot(kind="bar")

plt.title("Comparação entre Modelos Supervisionados")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(loc="lower right")

salvar_plot("comparacao_supervisionados", pastas["comparacao_modelos"])

# ==============================
# GRÁFICO 2 - KMEANS (ARI)
# ==============================
plt.figure(figsize=(6, 5))

ari_val = df_comp[df_comp["Modelo"] == "KMEANS"]["ARI"].values[0]

plt.bar(["KMeans"], [ari_val])
plt.title("Desempenho do K-Means (ARI)")
plt.ylabel("Adjusted Rand Index")

salvar_plot("kmeans_ari", pastas["comparacao_modelos"])

# ==============================
# COMPARAÇÃO COM BARRA DE ERRO
# ==============================

modelos = ["logistic", "tree", "knn"]
metricas = ["acc", "f1", "roc"]

nomes_metricas = {"acc": "Acurácia", "f1": "F1-score", "roc": "ROC-AUC"}

# preparar dados
medias = []
desvios = []

for modelo in modelos:
    media_modelo = []
    std_modelo = []

    for metrica in metricas:
        media_modelo.append(np.mean(resultados[modelo][metrica]))
        std_modelo.append(np.std(resultados[modelo][metrica]))

    medias.append(media_modelo)
    desvios.append(std_modelo)

medias = np.array(medias)
desvios = np.array(desvios)

x = np.arange(len(metricas))
width = 0.25

plt.figure(figsize=(10, 6))

for i, modelo in enumerate(modelos):
    plt.bar(
        x + i * width,
        medias[i],
        width,
        yerr=desvios[i],
        capsize=5,
        label=modelo.upper(),
    )

plt.xticks(x + width, [nomes_metricas[m] for m in metricas])
plt.ylabel("Score")
plt.title("Comparação de Modelos com Desvio Padrão")
plt.legend()

salvar_plot("comparacao_com_erro", pastas["comparacao_modelos"])

# ==============================
# MATRIZES DE CONFUSÃO (NOVO BLOCO)
# ==============================


def plot_cm(cm, titulo, nome):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        xticklabels=["Sem Doença", "Com Doença"],
        yticklabels=["Sem Doença", "Com Doença"],
    )
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.title(titulo)

    salvar_plot(nome, pastas["comparacao_modelos"])


# individuais
plot_cm(cm_log_total, "Logistic Regression", "cm_logistic")
plot_cm(cm_tree_total, "Decision Tree", "cm_tree")
plot_cm(cm_knn_total, "KNN", "cm_knn")


# ==============================
# MATRIZES LADO A LADO
# ==============================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

cms = [cm_log_total, cm_tree_total, cm_knn_total]
titulos = ["Logistic", "Tree", "KNN"]

for i in range(3):
    sns.heatmap(
        cms[i],
        annot=True,
        fmt=".0f",
        cmap="Blues",
        ax=axes[i],
        cbar=False,
        xticklabels=["0", "1"],
        yticklabels=["0", "1"],
    )
    axes[i].set_title(titulos[i])
    axes[i].set_xlabel("Previsto")
    axes[i].set_ylabel("Real")

plt.suptitle("Comparação das Matrizes de Confusão")

salvar_plot("comparacao_matrizes", pastas["comparacao_modelos"])

# ==============================
# ÁRVORE FINAL (TREINADA EM TODO O DATASET)
# ==============================

tree_final = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_final.fit(X, y)

# ==============================
# IMPORTÂNCIA DAS VARIÁVEIS
# ==============================
plt.figure(figsize=(10, 5))

importancias = pd.Series(tree_final.feature_importances_, index=X.columns)
importancias.sort_values(ascending=False).plot(kind="bar")

plt.title("Importância das Variáveis - Decision Tree")

salvar_plot("importancia_variaveis_tree", pastas["comparacao_modelos"])

plt.figure(figsize=(20, 10))

plot_tree(
    tree_final,
    feature_names=X.columns,
    class_names=["Sem Doença", "Com Doença"],
    filled=True,
    rounded=True,
    fontsize=8,
)

plt.title("Árvore de Decisão Final")

salvar_plot("arvore_decisao", pastas["comparacao_modelos"])

# ==============================
# 13. K-MEANS FINAL (VISUALIZAÇÃO)
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df["cluster"] = clusters

ari_global = adjusted_rand_score(y, clusters)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap="Set2",
    s=50,
    alpha=0.7,
)

centroides_pca = pca.transform(kmeans.cluster_centers_)

plt.scatter(
    centroides_pca[:, 0],
    centroides_pca[:, 1],
    c="red",
    marker="X",
    s=200,
    label="Centroides",
)

plt.title("Visualização do K-Means (PCA)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()

salvar_plot("kmeans_pca", pastas["comparacao_modelos"])

# ==============================
# 14. COMPARAÇÃO CLUSTER VS REAL
# ==============================
cm_clusters = confusion_matrix(y, clusters)

with open(resultados_path, "a") as f:
    f.write("\n\n=== KMEANS GLOBAL ===\n")
    f.write(f"ARI: {ari_global:.4f}\n")
    f.write("Matriz de Confusão:\n")
    f.write(str(cm_clusters))

print(f"\nResultados salvos em: {base_dir}")


# ==============================
# 15. TESTES NÃO PARAMÉTRICOS
# ==============================

print("\n=== TESTES ESTATÍSTICOS ===\n")

# ==============================
# FRIEDMAN (todos os modelos)
# ==============================
stat_friedman, p_friedman = friedmanchisquare(
    resultados["logistic"]["f1"],
    resultados["knn"]["f1"],
    resultados["tree"]["f1"]
)

print(f"Friedman Test -> p-value: {p_friedman:.4f}")

# ==============================
# WILCOXON (pares)
# ==============================

def teste_wilcoxon(modelo1, modelo2, nome1, nome2):
    stat, p = wilcoxon(modelo1, modelo2)
    print(f"{nome1} vs {nome2} -> p-value: {p:.4f}")

print("\nComparações par a par:")

teste_wilcoxon(resultados["logistic"]["f1"],
               resultados["knn"]["f1"],
               "Logistic", "KNN")

teste_wilcoxon(resultados["logistic"]["f1"],
               resultados["tree"]["f1"],
               "Logistic", "Tree")

teste_wilcoxon(resultados["knn"]["f1"],
               resultados["tree"]["f1"],
               "KNN", "Tree")

# ==============================
# BOXPLOT DAS MÉTRICAS
# ==============================
plt.figure(figsize=(8, 6))

dados_boxplot = [
    resultados["logistic"]["f1"],
    resultados["knn"]["f1"],
    resultados["tree"]["f1"]
]

sns.boxplot(data=dados_boxplot)

plt.xticks([0, 1, 2], ["Logistic", "KNN", "Tree"])
plt.ylabel("F1-score")
plt.title("Distribuição do F1-score por Modelo")

salvar_plot("boxplot_f1_modelos", pastas["comparacao_modelos"])