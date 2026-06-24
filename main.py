"""
Pipeline completo: EDA + TabPFN + SAE (overcomplete) + Decision Tree + TCAV
Baseado em: "Mechanistic Dynamic Interpretability for Tabular Foundation Models in Healthcare"

Mudanças em relação ao código original:
  1. SAE é OVERCOMPLETE (latent_dim = F * embedding_dim, não < embedding_dim)
     - Artigo usa F=1.5: embeddings de 192 dims → 288 fatores latentes
     - Objetivo: desempacotar superposição polisemântica, NÃO comprimir
  2. Tied weights no SAE (encoder/decoder compartilham W transposto)
  3. Normalização fitada SOMENTE no treino (sem data leakage)
  4. 4 splits independentes: discovery / cav_train / tcav_eval / held_out
  5. Decision Tree por fator com threshold no percentil 50 das ativações positivas
     Filtro: precisão ≥ 0.9 e recall ≥ 0.25
  6. TCAV com N=15 CAVs por conceito + distribuição nula + teste-t + FDR (BH)
     Filtro de efeito: |TCAV - 0.5| ≥ 0.1
  7. Interpretação via LASSO readout (feature_names reais, não emb_N)
"""

import gc
import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from scipy import stats
from sklearn.decomposition import DictionaryLearning
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from statsmodels.stats.multitest import multipletests
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ==============================
# CONFIGURAÇÃO
# ==============================

caminho_csv = "./Dataset/Processed_Data/dataset_unificado.csv"

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
base_dir = f"resultados_{timestamp}"

pastas = {
    "histogramas": os.path.join(base_dir, "histogramas"),
    "categoricos": os.path.join(base_dir, "categoricos"),
    "comparacoes": os.path.join(base_dir, "comparacoes"),
    "correlacao": os.path.join(base_dir, "correlacao"),
    "ml": os.path.join(base_dir, "machine_learning"),
}

for pasta in pastas.values():
    os.makedirs(pasta, exist_ok=True)

pasta_ml = pastas["ml"]

# ── SAE / TCAV ──────────────────────────────────────────────────────────────
EXPANSION_FACTOR = 1.5  # latent_dim = F * embedding_dim (overcomplete)
SAE_EPOCHS = 500
SAE_LR = 1e-3
SAE_SPARSITY = 0.05  # λ do artigo

DT_MAX_DEPTH = 5  # profundidade máxima da DT por fator
DT_PRECISION_MIN = 0.90  # artigo: precisão ≥ 0.90
DT_RECALL_MIN = 0.25  # artigo: recall   ≥ 0.25

N_TCAV_RUNS = 15  # artigo usa N=15
TCAV_EFFECT_MIN = 0.10  # |TCAV − 0.5| ≥ 0.1
FDR_ALPHA = 0.05
COHENS_D_MIN = 0.80  # artigo seção 4.2: effect size > 0.8 ("efeito grande")
RANDOM_STATE = 42

DL_N_COMPONENTS = 8  # artigo usa K=8 para Dictionary Learning (baseline)

# RANDOM_STATE: seed gerada aleatoriamente a cada execução do script.
# Para reproduzir um resultado específico depois, anote o valor impresso
# no console ("Seed desta execução: ...") e fixe-o manualmente aqui.
RANDOM_STATE = np.random.randint(0, 2**31 - 1)
print(f"Seed desta execução: {RANDOM_STATE}")
# ────────────────────────────────────────────────────────────────────────────

colunas_remover = [
    "id",
    "ccf",
    "name",
    "junk",
    "lvx1",
    "lvx2",
    "lvx3",
    "lvx4",
    "cathef",
    "ekgmo",
    "ekgday",
    "ekgyr",
    "cmo",
    "cday",
    "cyr",
    "restckm",
    "exerckm",
    "thalsev",
    "thalpul",
    "earlobe",
    "dummy",
]

colunas_vazamento = [
    "rcaprox",
    "rcadist",
    "ladprox",
    "laddist",
    "cxmain",
    "om1",
    "om2",
    "diag",
    "ramus",
    "lmt",
    "restef",
    "restwm",
    "exeref",
    "exerwm",
    "lvf",
]


def salvar_plot(nome, pasta):
    caminho = os.path.join(pasta, f"{nome}.png")
    plt.savefig(caminho, bbox_inches="tight")
    plt.close()


# ==============================
# CARREGAR E LIMPAR DADOS
# ==============================

print(f"Carregando: {caminho_csv}")
df = pd.read_csv(caminho_csv)
print("Dataset carregado:", df.shape)

coluna_origem = "dataset_origem" if "dataset_origem" in df.columns else None

for col in df.columns:
    if col != coluna_origem:
        df[col] = pd.to_numeric(df[col], errors="coerce")

colunas_vazias = [col for col in df.columns if df[col].isna().all()]
df = df.drop(columns=colunas_vazias)
df = df.drop(columns=[c for c in colunas_remover if c in df.columns])
df = df.drop(columns=[c for c in colunas_vazamento if c in df.columns], errors="ignore")


def atualizar_colunas(df):
    numericas = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if "num" in numericas:
        numericas.remove("num")
    categoricas = [c for c in numericas if df[c].nunique() < 10]
    numericas = [c for c in numericas if c not in categoricas]
    return numericas, categoricas


colunas_numericas, colunas_categoricas = atualizar_colunas(df)
df[colunas_numericas] = df[colunas_numericas].fillna(df[colunas_numericas].median())
for col in colunas_categoricas:
    moda = df[col].mode()
    if not moda.empty:
        df[col] = df[col].fillna(moda[0])

if "num" in df.columns:
    df["num"] = (df["num"] > 0).astype(int)

print("Shape após limpeza:", df.shape)

# ==============================
# PDF DESCRIÇÃO
# ==============================


def gerar_pdf_descricao(caminho_saida):
    if os.path.exists(caminho_saida):
        return
    styles = getSampleStyleSheet()
    elementos = []

    def add_titulo(t):
        elementos.extend([Paragraph(f"<b>{t}</b>", styles["Heading2"]), Spacer(1, 10)])

    def add_texto(t):
        elementos.extend([Paragraph(t, styles["BodyText"]), Spacer(1, 8)])

    descricoes = {
        "age": "Idade do paciente em anos.",
        "sex": "Sexo: 1=masculino, 0=feminino.",
        "cp": "Tipo de dor no peito: 1=angina típica, 2=atípica, 3=não cardíaca, 4=assintomático.",
        "trestbps": "Pressão arterial em repouso (mmHg).",
        "chol": "Colesterol (mg/dl).",
        "fbs": "Glicemia em jejum > 120 mg/dl. 1=sim.",
        "restecg": "ECG em repouso.",
        "thalach": "Frequência cardíaca máxima.",
        "exang": "Angina por esforço. 1=sim.",
        "oldpeak": "Depressão ST.",
        "slope": "Inclinação ST.",
        "ca": "Vasos afetados (0-3).",
        "thal": "Perfusão: 3=normal, 6=defeito fixo, 7=reversível.",
        "num": "Target: 0=sem doença, 1=com doença.",
    }
    add_titulo("Descrição do Dataset de Doença Cardíaca")
    for col, desc in descricoes.items():
        add_titulo(col)
        add_texto(desc)
    SimpleDocTemplate(caminho_saida).build(elementos)


pasta_desc = os.path.join(base_dir, "descricao")
os.makedirs(pasta_desc, exist_ok=True)
gerar_pdf_descricao(os.path.join(pasta_desc, "descricao_dataset.pdf"))

# ==============================
# EDA
# ==============================

colunas_numericas, colunas_categoricas = atualizar_colunas(df)

cols_existentes = [c for c in colunas_numericas if c in df.columns]
if cols_existentes:
    df[cols_existentes].hist(figsize=(18, 14))
    plt.suptitle("Distribuição das Variáveis Numéricas")
    salvar_plot("histogramas", pastas["histogramas"])

plt.figure(figsize=(6, 4))
sns.countplot(x="num", data=df)
plt.title("Distribuição da Doença")
salvar_plot("target", pastas["categoricos"])

if coluna_origem:
    plt.figure(figsize=(8, 5))
    sns.countplot(x="dataset_origem", hue="num", data=df)
    plt.title("Doença por Dataset")
    salvar_plot("doenca_por_origem", pastas["categoricos"])

for col in colunas_numericas:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="num", y=col, data=df)
    plt.title(f"{col} vs Doença")
    salvar_plot(f"{col}_boxplot", pastas["comparacoes"])

for col in colunas_categoricas:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, hue="num", data=df)
    plt.title(f"{col} vs Doença")
    salvar_plot(f"{col}_count", pastas["categoricos"])

df_numerico = df.select_dtypes(include=["int64", "float64"])
corr = df_numerico.corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Mapa de Correlação")
salvar_plot("correlacao_completa", pastas["correlacao"])

if "num" in df.columns:
    corr_target = corr["num"].sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    corr_target.drop("num").head(15).plot(kind="bar")
    plt.title("Top variáveis correlacionadas com doença")
    salvar_plot("top_corr", pastas["correlacao"])

# ==============================
# FEATURES / TARGET
# ==============================

X = df.drop(columns=["num", "dataset_origem"], errors="ignore")
y = df["num"]
feature_names = X.columns.tolist()

print(f"\nFeatures: {X.shape[1]}  |  Amostras: {X.shape[0]}")

# ==============================
# CROSS VALIDATION (TabPFN)
# ==============================

modelos = {"TabPFN": TabPFNClassifier()}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
resultados = {}

for nome, modelo in modelos.items():
    print(f"\n{'=' * 40}\nTreinando: {nome}\n{'=' * 40}")

    y_pred = cross_val_predict(modelo, X, y, cv=skf, n_jobs=1)
    f1 = f1_score(y, y_pred)
    resultados[nome] = f1
    print(f"F1-score: {f1:.4f}")

    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matriz de Confusão - {nome}")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    salvar_plot(f"matriz_confusao_{nome.replace(' ', '_')}", pasta_ml)

    report = classification_report(y, y_pred)
    print(report)
    with open(
        os.path.join(pasta_ml, f"classification_report_{nome.replace(' ', '_')}.txt"),
        "w",
    ) as f:
        f.write(report)

plt.figure(figsize=(8, 5))
sns.barplot(x=list(resultados.keys()), y=list(resultados.values()))
plt.ylim(0, 1)
plt.title("Comparação de Modelos (F1-score)")
plt.ylabel("F1-score")
for i, v in enumerate(resultados.values()):
    plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
salvar_plot("comparacao_f1_modelos", pasta_ml)

print("\n=== RESULTADOS FINAIS ===")
for nome, score in resultados.items():
    print(f"  {nome}: {score:.4f}")


# ============================================================
# BLOCO SAE + TCAV  (fiel ao artigo)
# ============================================================

print("\n" + "=" * 60)
print("PIPELINE DE INTERPRETABILIDADE (SAE + TCAV)")
print("=" * 60)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── 1. TREINO/TESTE com divisão temporal-like (estratificada) ───────────────
# Para dados sem coluna de tempo, usamos split estratificado 70/30.
# Se você tiver coluna de ano, substitua por divisão temporal explícita.

X_np = X.values.astype(np.float32)
y_np = y.values.astype(int)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_np,
    y_np,
    test_size=0.30,
    stratify=y_np,
    random_state=RANDOM_STATE,
)
print(f"\n  Treino: {len(y_train)}  |  Teste: {len(y_test)}")

# ── 2. TREINAR TABPFN ────────────────────────────────────────────────────────

print("\n[1/6] Treinando TabPFN para extração de embeddings...")

modelo_tabpfn = TabPFNClassifier(device=device)
modelo_tabpfn.fit(X_train_raw, y_train)


def extrair_embeddings(modelo, X_arr, is_test: bool = False):
    """
    Retorna array (N, D) com embeddings do TabPFN.

    O TabPFN distingue dois modos de extração:
      - data_source="train" → embeddings do contexto de treino (shape = n_train × D)
      - data_source="test"  → embeddings das amostras de consulta (shape = n_test × D)

    Quando is_test=True usamos "test", passando X_arr como X_test na chamada,
    mas o modelo precisa do X_train que já foi fornecido no .fit().
    Se get_embeddings não aceitar data_source, tentamos inferir pelo shape.
    """
    X_f = X_arr.astype(np.float32)

    with torch.no_grad():
        # Tenta com data_source explícito
        try:
            src = "test" if is_test else "train"
            emb = modelo.get_embeddings(X_f, data_source=src)
        except TypeError:
            # API antiga sem data_source
            emb = modelo.get_embeddings(X_f)

    if isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu().numpy()
    else:
        emb = np.asarray(emb)

    # Normalizar shape para (N_esperado, D)
    n_esperado = len(X_f)

    if emb.ndim == 3:
        # (estimators, N, D) → média dos estimadores
        emb = emb.mean(axis=0)

    if emb.ndim == 2 and emb.shape[0] != n_esperado:
        # Shape retornado não bate com n_esperado.
        # Isso ocorre quando data_source="test" retorna os embeddings do treino
        # ou vice-versa. Nesse caso tentamos o outro modo.
        try:
            src_alt = "train" if is_test else "test"
            with torch.no_grad():
                emb_alt = modelo.get_embeddings(X_f, data_source=src_alt)
            if isinstance(emb_alt, torch.Tensor):
                emb_alt = emb_alt.detach().cpu().numpy()
            if emb_alt.ndim == 3:
                emb_alt = emb_alt.mean(axis=0)
            if emb_alt.ndim == 2 and emb_alt.shape[0] == n_esperado:
                emb = emb_alt
        except Exception:
            pass

    if emb.ndim == 1:
        emb = np.repeat(emb.reshape(1, -1), n_esperado, axis=0)

    # Último recurso: se shape ainda não bate, repetir/truncar pela média
    if emb.shape[0] != n_esperado:
        print(
            f"  ⚠ Shape {emb.shape} ≠ {n_esperado} esperado — usando média por amostra"
        )
        emb = np.repeat(emb.mean(axis=0, keepdims=True), n_esperado, axis=0)

    return emb


train_emb_raw = extrair_embeddings(modelo_tabpfn, X_train_raw, is_test=False)
test_emb_raw = extrair_embeddings(modelo_tabpfn, X_test_raw, is_test=True)

EMBEDDING_DIM = train_emb_raw.shape[1]
LATENT_DIM = int(EXPANSION_FACTOR * EMBEDDING_DIM)

print(f"  Embedding dim : {EMBEDDING_DIM}")
print(f"  Latent dim SAE: {LATENT_DIM}  (F={EXPANSION_FACTOR}x → overcomplete)")

# ── 3. NORMALIZAÇÃO SEM LEAKAGE ──────────────────────────────────────────────

print("\n[2/6] Normalizando embeddings (scaler fitado só no treino)...")

scaler_emb = StandardScaler()
train_emb = scaler_emb.fit_transform(train_emb_raw)
test_emb = scaler_emb.transform(test_emb_raw)

# Salvar embeddings normalizados
pd.DataFrame(
    train_emb,
    columns=[f"emb_{i}" for i in range(EMBEDDING_DIM)],
).assign(target=y_train).to_csv(
    os.path.join(pasta_ml, "tabpfn_embeddings_train.csv"), index=False
)
pd.DataFrame(
    test_emb,
    columns=[f"emb_{i}" for i in range(EMBEDDING_DIM)],
).assign(target=y_test).to_csv(
    os.path.join(pasta_ml, "tabpfn_embeddings_test.csv"), index=False
)

# ============================================================
# COMPARAÇÃO: FEATURES ORIGINAIS vs. EMBEDDINGS TABPFN
# ============================================================
# Pergunta: o espaço latente do TabPFN carrega informação preditiva extra
# em relação às features originais, quando usado por OUTROS modelos?
#
# Treina RL, RF, XGBoost e TabPFN duas vezes:
#   (A) nas 38 features originais (X_train_raw / X_test_raw)
#   (B) nos embeddings normalizados do TabPFN (train_emb / test_emb, 512 dims)
# Se RL/RF/XGBoost melhorarem no cenário B, o TabPFN está funcionando como
# um bom extrator de features não-lineares para modelos mais simples.

print("\n" + "=" * 60)
print("COMPARAÇÃO: FEATURES ORIGINAIS vs. EMBEDDINGS TABPFN")
print("=" * 60)

# RL precisa de features em escala comparável; RF e XGBoost não precisam,
# mas normalizar não piora e mantém o código simples para os 3.
scaler_orig = StandardScaler()
X_train_orig_n = scaler_orig.fit_transform(X_train_raw)
X_test_orig_n = scaler_orig.transform(X_test_raw)


def avaliar_modelos(
    X_tr,
    y_tr,
    X_te,
    y_te,
    modelo_tabpfn_treinado=None,
    X_te_tabpfn=None,
    nome_cenario="",
):
    """
    Treina RL, RF, XGBoost (e opcionalmente reusa um TabPFN já treinado)
    no mesmo split treino/teste, retornando F1, acurácia e relatório por modelo.

    X_te_tabpfn: dados de teste para o TabPFN especificamente. O TabPFN foi
    treinado em dados BRUTOS (sem normalização) — se receber X_te normalizado
    (escala/distribuição diferente da vista no fit), a predição degenera e
    colapsa para uma única classe (F1=0). Por isso o TabPFN sempre recebe seu
    próprio X_te não normalizado, independente do X_te usado pelos outros 3.
    """
    modelos_cenario = {
        "Logistic Regression": LogisticRegression(
            max_iter=5000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }

    resultados_cenario = {}

    for nome_m, modelo_m in modelos_cenario.items():
        modelo_m.fit(X_tr, y_tr)
        y_pred_m = modelo_m.predict(X_te)
        f1_m = f1_score(y_te, y_pred_m)
        acc_m = (y_pred_m == y_te).mean()
        resultados_cenario[nome_m] = {"f1": f1_m, "acc": acc_m}
        print(f"    {nome_m:<22} | F1={f1_m:.4f} | Acc={acc_m:.4f}")

    # TabPFN: reaproveita o modelo já treinado (evita treinar de novo).
    # Usa X_te_tabpfn (dados brutos) em vez de X_te (pode estar normalizado).
    if modelo_tabpfn_treinado is not None:
        X_te_para_tabpfn = X_te_tabpfn if X_te_tabpfn is not None else X_te
        y_pred_tabpfn = modelo_tabpfn_treinado.predict(X_te_para_tabpfn)
        f1_tabpfn = f1_score(y_te, y_pred_tabpfn)
        acc_tabpfn = (y_pred_tabpfn == y_te).mean()
        resultados_cenario["TabPFN"] = {"f1": f1_tabpfn, "acc": acc_tabpfn}
        print(f"    {'TabPFN':<22} | F1={f1_tabpfn:.4f} | Acc={acc_tabpfn:.4f}")

    return resultados_cenario


print("\n[A] Treinando nas FEATURES ORIGINAIS (38 dims, normalizadas)...")
resultados_originais = avaliar_modelos(
    X_train_orig_n,
    y_train,
    X_test_orig_n,
    y_test,
    modelo_tabpfn_treinado=modelo_tabpfn,  # já treinado em X_train_raw
    X_te_tabpfn=X_test_raw,  # TabPFN recebe dados BRUTOS, não normalizados
    nome_cenario="originais",
)


print(f"\n[B] Treinando nos EMBEDDINGS TABPFN ({EMBEDDING_DIM} dims, normalizados)...")
# Para o TabPFN no cenário B, é preciso um modelo NOVO treinado a partir dos
# embeddings como se fossem features de entrada — o TabPFN original não
# "recebe" embeddings como X, então aqui ele funciona apenas como gerador
# de representação para os outros 3 modelos. Reportamos o TabPFN original
# (cenário A) como referência na mesma linha para comparação visual.
resultados_embedding = avaliar_modelos(
    train_emb,
    y_train,
    test_emb,
    y_test,
    modelo_tabpfn_treinado=None,  # TabPFN não é re-treinado em seu próprio embedding
    nome_cenario="embeddings",
)
# Adiciona o TabPFN original (cenário A) como linha de referência fixa
resultados_embedding["TabPFN (referência)"] = resultados_originais["TabPFN"]

# ── Tabela comparativa consolidada ───────────────────────────────────────────

linhas_comparacao = []
for nome_m in ["Logistic Regression", "Random Forest", "XGBoost"]:
    linhas_comparacao.append(
        {
            "Modelo": nome_m,
            "F1 (Original)": round(resultados_originais[nome_m]["f1"], 4),
            "F1 (Embedding)": round(resultados_embedding[nome_m]["f1"], 4),
            "Δ F1": round(
                resultados_embedding[nome_m]["f1"] - resultados_originais[nome_m]["f1"],
                4,
            ),
            "Acc (Original)": round(resultados_originais[nome_m]["acc"], 4),
            "Acc (Embedding)": round(resultados_embedding[nome_m]["acc"], 4),
        }
    )
linhas_comparacao.append(
    {
        "Modelo": "TabPFN",
        "F1 (Original)": round(resultados_originais["TabPFN"]["f1"], 4),
        "F1 (Embedding)": "—",  # TabPFN não roda sobre seu próprio embedding
        "Δ F1": "—",
        "Acc (Original)": round(resultados_originais["TabPFN"]["acc"], 4),
        "Acc (Embedding)": "—",
    }
)

df_comparacao_modelos = pd.DataFrame(linhas_comparacao)
df_comparacao_modelos.to_csv(
    os.path.join(pasta_ml, "comparacao_modelos_original_vs_embedding.csv"), index=False
)

print("\n  Tabela comparativa (Original vs. Embedding TabPFN):")
print(df_comparacao_modelos.to_string(index=False))

# ── Plot comparativo ─────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))
nomes_plot = ["Logistic\nRegression", "Random\nForest", "XGBoost", "TabPFN"]
f1_orig_plot = [
    resultados_originais["Logistic Regression"]["f1"],
    resultados_originais["Random Forest"]["f1"],
    resultados_originais["XGBoost"]["f1"],
    resultados_originais["TabPFN"]["f1"],
]
f1_emb_plot = [
    resultados_embedding["Logistic Regression"]["f1"],
    resultados_embedding["Random Forest"]["f1"],
    resultados_embedding["XGBoost"]["f1"],
    None,  # TabPFN não tem barra de embedding
]

x_pos = np.arange(len(nomes_plot))
largura = 0.35

ax.bar(
    x_pos - largura / 2,
    f1_orig_plot,
    largura,
    label="Features Originais (38 dims)",
    color="#2563EB",
)
f1_emb_plot_seguro = [v if v is not None else 0 for v in f1_emb_plot]
barras_emb = ax.bar(
    x_pos + largura / 2,
    f1_emb_plot_seguro,
    largura,
    label=f"Embedding TabPFN ({EMBEDDING_DIM} dims)",
    color="#7C3AED",
)

# Marcar visualmente que TabPFN não tem barra de embedding
barras_emb[-1].set_alpha(0.15)
barras_emb[-1].set_hatch("//")

ax.set_xticks(x_pos)
ax.set_xticklabels(nomes_plot)
ax.set_ylabel("F1-score")
ax.set_ylim(0, 1)
ax.set_title("Comparação de Modelos: Features Originais vs. Embedding TabPFN")
ax.legend()

for i, (v_orig, v_emb) in enumerate(zip(f1_orig_plot, f1_emb_plot)):
    ax.text(i - largura / 2, v_orig + 0.015, f"{v_orig:.3f}", ha="center", fontsize=9)
    if v_emb is not None:
        ax.text(i + largura / 2, v_emb + 0.015, f"{v_emb:.3f}", ha="center", fontsize=9)

salvar_plot("comparacao_original_vs_embedding", pasta_ml)

print(f"\n  Interpretação:")
melhoraram = [
    nome_m
    for nome_m in ["Logistic Regression", "Random Forest", "XGBoost"]
    if resultados_embedding[nome_m]["f1"] > resultados_originais[nome_m]["f1"]
]
if melhoraram:
    print(f"  Modelos que melhoraram com o embedding TabPFN: {', '.join(melhoraram)}")
    print(f"  → O espaço latente do TabPFN carrega informação preditiva extra")
    print(f"    além das features originais para esses modelos.")
else:
    print(f"  Nenhum modelo melhorou com o embedding — as features originais já")
    print(f"  capturam a informação relevante tão bem quanto o embedding bruto.")

# ── 4. 4 SPLITS INDEPENDENTES DO TESTE ──────────────────────────────────────
# discovery → treinar SAE e DTs
# cav_train → treinar CAVs
# tcav_eval → avaliar TCAV scores
# held_out  → Random Forest final

idx_all = np.arange(len(y_test))

idx_disc, idx_rest = train_test_split(
    idx_all, test_size=0.67, random_state=RANDOM_STATE, stratify=y_test
)
idx_cav_train, idx_eval_hold = train_test_split(
    idx_rest, test_size=0.5, random_state=RANDOM_STATE, stratify=y_test[idx_rest]
)
idx_tcav_eval, idx_held_out = train_test_split(
    idx_eval_hold,
    test_size=0.5,
    random_state=RANDOM_STATE,
    stratify=y_test[idx_eval_hold],
)

emb_disc = test_emb[idx_disc]
emb_cav_train = test_emb[idx_cav_train]
emb_tcav_eval = test_emb[idx_tcav_eval]
emb_held_out = test_emb[idx_held_out]

X_disc_orig = X_test_raw[idx_disc]
X_cav_orig = X_test_raw[idx_cav_train]
X_held_orig = X_test_raw[idx_held_out]

y_disc = y_test[idx_disc]
y_held_out_labels = y_test[idx_held_out]

print(f"  Discovery : {len(idx_disc):4d} amostras")
print(f"  CAV Train : {len(idx_cav_train):4d} amostras")
print(f"  TCAV Eval : {len(idx_tcav_eval):4d} amostras")
print(f"  Held-Out  : {len(idx_held_out):4d} amostras")


# ── 5. SAE OVERCOMPLETE COM TIED WEIGHTS ────────────────────────────────────

print(f"\n[3/6] Treinando SAE overcomplete ({EMBEDDING_DIM}→{LATENT_DIM})...")


class SparseAutoencoder(nn.Module):
    """
    SAE com tied weights (artigo seção 3.3):
        h    = ReLU(W_e · z  + b_e)      encoder
        z_hat = W_e^T · h + b_d          decoder (pesos amarrados)
    As colunas de W_e^T são os vetores-direção dos conceitos no espaço latente.
    """

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.W_e = nn.Parameter(torch.randn(latent_dim, input_dim) * 0.01)
        self.b_e = nn.Parameter(torch.zeros(latent_dim))
        self.b_d = nn.Parameter(torch.zeros(input_dim))

    def encode(self, z):
        return torch.relu(z @ self.W_e.T + self.b_e)

    def decode(self, h):
        return h @ self.W_e + self.b_d  # tied: W_d = W_e

    def forward(self, z):
        h = self.encode(z)
        z_hat = self.decode(h)
        return z_hat, h


X_sae = torch.tensor(emb_disc, dtype=torch.float32, device=device)

sae = SparseAutoencoder(EMBEDDING_DIM, LATENT_DIM).to(device)
opt = optim.Adam(sae.parameters(), lr=SAE_LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SAE_EPOCHS)

hist = {"loss": [], "mse": [], "sparse": []}

for epoch in range(SAE_EPOCHS):
    z_hat, h = sae(X_sae)
    mse = ((z_hat - X_sae) ** 2).mean()
    sparse = h.abs().mean()
    loss = mse + SAE_SPARSITY * sparse

    opt.zero_grad()
    loss.backward()
    opt.step()
    scheduler.step()

    hist["loss"].append(loss.item())
    hist["mse"].append(mse.item())
    hist["sparse"].append(sparse.item())

    if epoch % 100 == 0:
        nz = (h.detach().abs() < 1e-5).float().mean().item()
        av = (h.detach().abs() >= 1e-5).float().sum(dim=1).mean().item()
        print(
            f"  Epoch {epoch:4d} | loss={loss.item():.5f} | mse={mse.item():.5f} "
            f"| near-zero={nz:.1%} | ativo/amostra={av:.1f}"
        )

# Métricas de decomposição (artigo Tabela 1)
with torch.no_grad():
    _, h_disc = sae(X_sae)
    h_disc_np = h_disc.cpu().numpy()

near_zero_rate = (np.abs(h_disc_np) < 1e-5).mean()
active_per_sample = (np.abs(h_disc_np) >= 1e-5).sum(axis=1).mean()
dead_factors = np.where(h_disc_np.max(axis=0) < 1e-5)[0]
active_factors = np.where(h_disc_np.max(axis=0) >= 1e-5)[0]

# Ortogonalidade
W_d = sae.W_e.detach().cpu().numpy().T  # (input_dim, latent_dim)
norms = np.linalg.norm(W_d, axis=0, keepdims=True)
norms[norms == 0] = 1
W_norm = W_d / norms
cos_sim = np.abs(W_norm.T @ W_norm)
np.fill_diagonal(cos_sim, 0)
high_sim_pairs = (cos_sim > 0.5).sum() // 2

print(f"\n  Métricas SAE:")
print(f"  Near-zero rate  : {near_zero_rate:.1%}")
print(f"  Ativos/amostra  : {active_per_sample:.1f} / {LATENT_DIM}")
print(f"  Fatores mortos  : {len(dead_factors)} / {LATENT_DIM}")
print(f"  Fatores ativos  : {len(active_factors)}")
print(f"  Pares cos>0.5   : {high_sim_pairs}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(hist["mse"], label="MSE", color="#2563EB")
plt.plot(hist["sparse"], label="Sparsity", color="#DC2626", alpha=0.7)
plt.yscale("log")
plt.xlabel("Epoch")
plt.title("Componentes do Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(hist["loss"], color="#7C3AED")
plt.yscale("log")
plt.xlabel("Epoch")
plt.title("Loss Total")
salvar_plot("sae_loss", pasta_ml)


# helper: obter ativações SAE para qualquer array de embeddings
def get_activations(emb_np):
    t = torch.tensor(emb_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        _, h = sae(t)
    return h.cpu().numpy()


acts_disc = h_disc_np
acts_cav_train = get_activations(emb_cav_train)
acts_tcav_eval = get_activations(emb_tcav_eval)
acts_held_out = get_activations(emb_held_out)


# ============================================================
# DICTIONARY LEARNING — Baseline Linear (artigo seção 3.3 e Tabela 1)
# ============================================================
# O artigo compara SAE (não-linear, overcomplete) contra Dictionary
# Learning (linear, compacto, K=8) para justificar a escolha do SAE.
#
# DL resolve: min_Φ,c  Σ ||z_i - Φc_i||² + λ||c_i||₁   s.t. ||φ_k|| ≤ 1
# usando sklearn.decomposition.DictionaryLearning

print(
    f"\n[3b/6] Treinando Dictionary Learning (K={DL_N_COMPONENTS}) para comparação..."
)

from sklearn.decomposition import DictionaryLearning

dl = DictionaryLearning(
    n_components=DL_N_COMPONENTS,
    alpha=1.0,  # equivalente ao λ do artigo
    max_iter=500,
    random_state=RANDOM_STATE,
    transform_algorithm="lasso_lars",  # produz códigos esparsos
)

# Treina no mesmo split de discovery usado pelo SAE (comparação justa)
codigos_dl = dl.fit_transform(emb_disc)  # (n_disc, K) — códigos esparsos c_i
dicionario_dl = dl.components_  # (K, D)     — átomos Φ

# Reconstrução: z_hat = c @ Φ
emb_disc_reconstruido_dl = codigos_dl @ dicionario_dl


def calcular_metricas_decomposicao(z_original, z_reconstruido, ativacoes, direcoes):
    """
    Réplica das 4 métricas da Tabela 1 do artigo:
      - MSE de reconstrução
      - Sparsity (% de ativações |.| < 1e-5)
      - Active/Sample (unidades ativas por amostra, em média)
      - Direction Similarity (% de pares de direções com cos_sim > 0.5)
    """
    mse = float(np.mean((z_original - z_reconstruido) ** 2))

    near_zero = float((np.abs(ativacoes) < 1e-5).mean())
    active_per_sample_ = float((np.abs(ativacoes) >= 1e-5).sum(axis=1).mean())

    norms = np.linalg.norm(direcoes, axis=1, keepdims=True)
    norms[norms == 0] = 1
    direcoes_norm = direcoes / norms
    cos_sim = np.abs(direcoes_norm @ direcoes_norm.T)
    np.fill_diagonal(cos_sim, 0)
    n_pares = cos_sim.shape[0] * (cos_sim.shape[0] - 1) / 2
    pct_pares_similares = (
        float((cos_sim > 0.5).sum() / 2 / n_pares) if n_pares > 0 else 0.0
    )

    return {
        "mse": mse,
        "sparsity": near_zero,
        "active_per_sample": active_per_sample_,
        "n_units": ativacoes.shape[1],
        "pct_direction_sim": pct_pares_similares,
    }


metricas_dl = calcular_metricas_decomposicao(
    z_original=emb_disc,
    z_reconstruido=emb_disc_reconstruido_dl,
    ativacoes=codigos_dl,
    direcoes=dicionario_dl,
)

# MSE do SAE no mesmo split (discovery), para comparação justa
with torch.no_grad():
    z_hat_sae_disc, _ = sae(X_sae)
    mse_sae = float(((z_hat_sae_disc - X_sae) ** 2).mean().cpu())

metricas_sae = {
    "mse": mse_sae,
    "sparsity": near_zero_rate,
    "active_per_sample": active_per_sample,
    "n_units": LATENT_DIM,
    "pct_direction_sim": (
        float(high_sim_pairs / (LATENT_DIM * (LATENT_DIM - 1) / 2))
        if LATENT_DIM > 1
        else 0.0
    ),
}

df_comparacao_decomposicao = pd.DataFrame(
    [
        {
            "Method": "Dictionary Learning",
            "MSE": round(metricas_dl["mse"], 4),
            "Sparsity": f"{metricas_dl['sparsity']:.1%}",
            "Active/Sample": f"{metricas_dl['active_per_sample']:.1f} / {metricas_dl['n_units']}",
            "Direction Sim. (>0.5)": f"{metricas_dl['pct_direction_sim']:.1%}",
        },
        {
            "Method": "Sparse Autoencoder",
            "MSE": round(metricas_sae["mse"], 4),
            "Sparsity": f"{metricas_sae['sparsity']:.1%}",
            "Active/Sample": f"{metricas_sae['active_per_sample']:.1f} / {metricas_sae['n_units']}",
            "Direction Sim. (>0.5)": f"{metricas_sae['pct_direction_sim']:.1%}",
        },
    ]
)

print("\n  Tabela 1 — Qualidade da Decomposição (DL vs SAE):")
print(df_comparacao_decomposicao.to_string(index=False))

df_comparacao_decomposicao.to_csv(
    os.path.join(pasta_ml, "comparacao_dl_vs_sae.csv"), index=False
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
metodos = ["Dictionary\nLearning", "Sparse\nAutoencoder"]
cores_metodo = ["#94A3B8", "#7C3AED"]

axes[0].bar(metodos, [metricas_dl["mse"], metricas_sae["mse"]], color=cores_metodo)
axes[0].set_yscale("log")
axes[0].set_title("MSE de Reconstrução\n(menor é melhor)")
axes[0].set_ylabel("MSE (log)")

axes[1].bar(
    metodos,
    [metricas_dl["sparsity"] * 100, metricas_sae["sparsity"] * 100],
    color=cores_metodo,
)
axes[1].set_title("Sparsity\n(% ativações ≈ 0)")
axes[1].set_ylabel("%")
axes[1].set_ylim(0, 100)

axes[2].bar(
    metodos,
    [metricas_dl["pct_direction_sim"] * 100, metricas_sae["pct_direction_sim"] * 100],
    color=cores_metodo,
)
axes[2].set_title("Pares com cos_sim > 0.5\n(menor é melhor)")
axes[2].set_ylabel("%")

salvar_plot("comparacao_dl_vs_sae", pasta_ml)

print(f"\n  Interpretação:")
if metricas_sae["mse"] < metricas_dl["mse"]:
    reducao = (1 - metricas_sae["mse"] / metricas_dl["mse"]) * 100
    print(
        f"  SAE reduz o erro de reconstrução em {reducao:.1f}% vs Dictionary Learning"
    )
if metricas_sae["pct_direction_sim"] < metricas_dl["pct_direction_sim"]:
    print(f"  SAE produz direções mais ortogonais (menos redundância entre conceitos)")


# ── 6. DECISION TREE POR FATOR ───────────────────────────────────────────────
# Threshold = percentil 50 das ativações positivas (artigo seção 3.5)
# Retém fator somente se precisão ≥ 0.9 E recall ≥ 0.25

print(
    f"\n[4/6] Decision Trees por fator SAE (P≥{DT_PRECISION_MIN}, R≥{DT_RECALL_MIN})..."
)

conceitos_candidatos = []


def extrair_regra_completa(dt: DecisionTreeClassifier, feature_names: list) -> str:
    """
    Percorre a árvore do nó raiz até a folha com maior precisão para a classe
    positiva (1), retornando a conjunção COMPLETA de condições no caminho
    — não apenas o primeiro split. Formato: 'cond1 AND cond2 AND cond3 ...'
    Réplica do estilo de regra usado na Tabela 2 do artigo (ex.:
    "10.5 < EVENT_C1DIALISE_HD <= 13.5 AND EVENT_c5TX_EXTX <= 0.5").
    """
    tree = dt.tree_

    # Identificar a folha (classe 1) com maior número de amostras positivas
    # entre as folhas que predizem a classe positiva.
    leaf_ids = [i for i in range(tree.node_count) if tree.children_left[i] == -1]
    melhor_leaf, melhor_score = None, -1
    for leaf in leaf_ids:
        valores = tree.value[leaf][0]
        classe_pred = int(np.argmax(valores))
        if classe_pred == 1:
            score = valores[
                1
            ]  # peso (amostras, ponderado por class_weight) da classe 1
            if score > melhor_score:
                melhor_score, melhor_leaf = score, leaf

    if melhor_leaf is None:
        return "nenhuma folha prediz classe positiva"

    # Reconstruir o caminho da raiz até essa folha
    def caminho_para_no(node_id, caminho=()):
        if node_id == melhor_leaf:
            return caminho
        esq, dir_ = tree.children_left[node_id], tree.children_right[node_id]
        if esq != -1:
            r = caminho_para_no(esq, caminho + ((node_id, "esq"),))
            if r is not None:
                return r
        if dir_ != -1:
            r = caminho_para_no(dir_, caminho + ((node_id, "dir"),))
            if r is not None:
                return r
        return None

    caminho = caminho_para_no(0)
    if caminho is None:
        return "caminho não encontrado"

    # Montar condições, agregando limites min/max por feature (ex: "10.5 < X <= 13.5")
    limites = {}  # feature_idx -> [lower, upper]
    for node_id, direcao in caminho:
        feat_idx = tree.feature[node_id]
        thresh = tree.threshold[node_id]
        if feat_idx not in limites:
            limites[feat_idx] = [-np.inf, np.inf]
        if direcao == "esq":  # feature <= thresh
            limites[feat_idx][1] = min(limites[feat_idx][1], thresh)
        else:  # feature > thresh
            limites[feat_idx][0] = max(limites[feat_idx][0], thresh)

    condicoes = []
    for feat_idx, (lo, hi) in limites.items():
        nome = feature_names[feat_idx]
        if lo == -np.inf:
            condicoes.append(f"{nome} <= {hi:.2f}")
        elif hi == np.inf:
            condicoes.append(f"{nome} > {lo:.2f}")
        else:
            condicoes.append(f"{lo:.2f} < {nome} <= {hi:.2f}")

    return " AND ".join(condicoes)


for k in active_factors:
    ativ_k = acts_disc[:, k]
    positivas = ativ_k[ativ_k > 0]

    if len(positivas) < 10:
        continue

    threshold_k = np.percentile(positivas, 50)
    y_k = (ativ_k > threshold_k).astype(int)

    if y_k.sum() < 5 or (1 - y_k).sum() < 5:
        continue

    dt = DecisionTreeClassifier(
        max_depth=DT_MAX_DEPTH,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    dt.fit(X_disc_orig, y_k)
    y_pred_k = dt.predict(X_disc_orig)

    tp = int(((y_pred_k == 1) & (y_k == 1)).sum())
    fp = int(((y_pred_k == 1) & (y_k == 0)).sum())
    fn = int(((y_pred_k == 0) & (y_k == 1)).sum())

    precisao = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)

    if precisao >= DT_PRECISION_MIN and recall >= DT_RECALL_MIN:
        regra_completa = extrair_regra_completa(dt, feature_names)

        conceitos_candidatos.append(
            {
                "fator": k,
                "precisao": round(precisao, 3),
                "recall": round(recall, 3),
                "threshold_k": threshold_k,
                "regra": regra_completa,
                "dt_model": dt,
            }
        )

print(
    f"  Candidatos após filtro DT: {len(conceitos_candidatos)} / {len(active_factors)} ativos"
)

if len(conceitos_candidatos) == 0:
    print("\n  ⚠ Nenhum conceito passou o filtro da Decision Tree.")
    print("  Sugestões: reduzir DT_PRECISION_MIN para 0.80 ou DT_RECALL_MIN para 0.15")


# ── 7. TCAV COM VALIDAÇÃO ESTATÍSTICA ───────────────────────────────────────

print(f"\n[5/6] Calculando TCAV (N={N_TCAV_RUNS} runs + FDR-BH)...")


def treinar_cav(pos_emb: np.ndarray, neg_emb: np.ndarray) -> np.ndarray:
    """CAV = vetor normal da fronteira logística entre positivos e negativos."""
    X_cav = np.vstack([pos_emb, neg_emb])
    y_cav = np.array([1] * len(pos_emb) + [0] * len(neg_emb))
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    clf.fit(X_cav, y_cav)
    v = clf.coef_[0]
    return v / (np.linalg.norm(v) + 1e-9)


def calcular_scores_tcav(cavs: list, emb_eval: np.ndarray) -> list:
    """Sensibilidade direcional: fração de amostras com produto interno > 0."""
    z_c = emb_eval - emb_eval.mean(axis=0)
    return [float((z_c @ cav > 0).mean()) for cav in cavs]


# Distribuição nula: CAV random vs random
rng = np.random.default_rng(RANDOM_STATE)
null_dist = []
n_eval = len(emb_tcav_eval)
for _ in range(N_TCAV_RUNS):
    idx_a = rng.choice(n_eval, size=min(20, n_eval // 2), replace=False)
    idx_b = np.setdiff1d(np.arange(n_eval), idx_a)[:20]
    cav_null = treinar_cav(emb_tcav_eval[idx_a], emb_tcav_eval[idx_b])
    z_c = emb_tcav_eval - emb_tcav_eval.mean(axis=0)
    null_dist.append(float((z_c @ cav_null > 0).mean()))

resultados_tcav = []

for conceito in conceitos_candidatos:
    k = conceito["fator"]
    dt_k = conceito["dt_model"]

    # Positivos no split cav_train (via regra da DT)
    pos_mask_cav = dt_k.predict(X_cav_orig) == 1
    if pos_mask_cav.sum() < 5:
        continue

    pos_emb_cav = emb_cav_train[pos_mask_cav]
    neg_pool = np.where(~pos_mask_cav)[0]

    # N=15 CAVs com datasets negativos diferentes
    cavs_k = []
    for _ in range(N_TCAV_RUNS):
        n_neg = min(len(pos_emb_cav), len(neg_pool))
        neg_idx = rng.choice(neg_pool, size=n_neg, replace=False)
        cavs_k.append(treinar_cav(pos_emb_cav, emb_cav_train[neg_idx]))

    scores_k = calcular_scores_tcav(cavs_k, emb_tcav_eval)
    tcav_mean = np.mean(scores_k)
    tcav_std = np.std(scores_k)

    t_stat, p_val = stats.ttest_ind(scores_k, null_dist)

    # ── Cohen's d (effect size) — artigo seção 4.2 ───────────────────────────
    # Mede a magnitude da diferença entre os TCAV scores do conceito e a
    # distribuição nula, em unidades de desvio-padrão conjunto (pooled SD).
    # Complementa o p-value: um conceito pode ser estatisticamente significativo
    # (p baixo) mas ter efeito pequeno se N for grande — Cohen's d não depende
    # do tamanho da amostra, só da separação real entre as distribuições.
    #   d = (média_grupo1 - média_grupo2) / SD_conjunto
    #   SD_conjunto = sqrt( ((n1-1)*s1² + (n2-1)*s2²) / (n1+n2-2) )
    # Artigo exige d > 0.8 (convencionalmente "efeito grande").
    n1, n2 = len(scores_k), len(null_dist)
    var1, var2 = np.var(scores_k, ddof=1), np.var(null_dist, ddof=1)
    sd_pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohens_d = float((tcav_mean - np.mean(null_dist)) / (sd_pooled + 1e-9))

    resultados_tcav.append(
        {
            "fator": k,
            "precisao": conceito["precisao"],
            "recall": conceito["recall"],
            "regra": conceito["regra"],
            "tcav_mean": round(tcav_mean, 4),
            "tcav_std": round(tcav_std, 4),
            "p_valor": p_val,
            "cohens_d": round(cohens_d, 4),
            "desvio_chance": round(abs(tcav_mean - 0.5), 4),
            "cavs": cavs_k,
            "dt_model": conceito["dt_model"],
            "threshold_k": conceito["threshold_k"],
        }
    )

# Correção FDR (Benjamini-Hochberg)
conceitos_finais = []
if resultados_tcav:
    p_vals = [r["p_valor"] for r in resultados_tcav]
    _, p_adj, _, _ = multipletests(p_vals, alpha=FDR_ALPHA, method="fdr_bh")
    for i, r in enumerate(resultados_tcav):
        r["p_adj"] = round(p_adj[i], 5)

    # Filtro final: significância (FDR) + efeito mínimo (desvio chance) +
    # effect size formal (Cohen's d) — os 3 critérios do artigo combinados.
    conceitos_finais = [
        r
        for r in resultados_tcav
        if r["p_adj"] < FDR_ALPHA
        and r["desvio_chance"] >= TCAV_EFFECT_MIN
        and abs(r["cohens_d"]) >= COHENS_D_MIN
    ]

print(
    f"  Aprovados (p_adj<{FDR_ALPHA}, |TCAV-0.5|≥{TCAV_EFFECT_MIN}, |d|≥{COHENS_D_MIN}): {len(conceitos_finais)}"
)

if not conceitos_finais:
    print("\n  ⚠ Nenhum conceito passou o filtro TCAV.")
    print(
        "  Tente: reduzir TCAV_EFFECT_MIN para 0.05, COHENS_D_MIN para 0.5, ou DT_RECALL_MIN para 0.15"
    )


# ── 8. RANDOM FOREST + LASSO NOS CONCEITOS FINAIS ───────────────────────────

if conceitos_finais:
    print(
        f"\n[6/6] Random Forest + LASSO readout nos {len(conceitos_finais)} conceitos finais..."
    )

    fatores_idx = [c["fator"] for c in conceitos_finais]
    acts_held_final = acts_held_out[:, fatores_idx]

    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    rf.fit(acts_held_final, y_held_out_labels)

    for i, c in enumerate(conceitos_finais):
        c["importancia_rf"] = float(rf.feature_importances_[i])

    # LASSO readout: mapeia cada fator para as features originais
    for c in conceitos_finais:
        k = c["fator"]
        ativ = acts_held_out[:, k]
        lasso = Lasso(alpha=0.01, max_iter=5000)
        lasso.fit(X_held_orig, ativ)
        coef_abs = np.abs(lasso.coef_)
        top5_idx = np.argsort(coef_abs)[-5:][::-1]
        c["top5_features"] = [feature_names[j] for j in top5_idx]
        c["top5_coef"] = [float(lasso.coef_[j]) for j in top5_idx]

    # ════════════════════════════════════════════════════════════════════════
    # DESTRUCTION TEST E SUFFICIENCY TEST (artigo seção 3.4 / eq. 8-9)
    # ════════════════════════════════════════════════════════════════════════
    # O artigo projeta o embedding no espaço nulo do CAV (destruction) ou só
    # na direção do CAV (sufficiency), e mede como a predição do modelo muda.
    #
    # Limitação prática: o TabPFN não expõe um "decoder" que aceite embeddings
    # diretamente — a predição sempre depende de um forward pass completo com
    # o contexto de treino. Por isso, a perturbação aqui é feita no espaço de
    # ENTRADA ORIGINAL (X), usando a direção do CAV mapeada de volta para as
    # features originais via os coeficientes do LASSO já calculados acima.
    # Isso preserva a lógica de necessidade/suficiência do artigo (eq. 8-9),
    # adaptada para um modelo sem acesso direto ao espaço de embedding.

    print(f"\n[7/6] Testes de ablação (Destruction + Sufficiency)...")

    def construir_direcao_em_X(c, feature_names_list):
        """
        Constrói um vetor direção no espaço de X (features originais) a partir
        dos coeficientes do LASSO que mapeiam o conceito SAE para X. Isso é o
        equivalente prático ao CAV v_k do artigo, mas no espaço de entrada.
        """
        k = c["fator"]
        ativ = acts_held_out[:, k]
        lasso_full = Lasso(alpha=0.01, max_iter=5000)
        lasso_full.fit(X_held_orig, ativ)
        v = lasso_full.coef_.copy()
        norma = np.linalg.norm(v)
        if norma < 1e-9:
            return None
        return v / norma

    def predizer_proba(modelo, X_arr):
        """Probabilidade da classe positiva (1) para um array de amostras."""
        X_f = np.atleast_2d(X_arr).astype(np.float32)
        proba = modelo.predict_proba(X_f)
        return proba[:, 1]

    # Amostras onde o conceito está ATIVO no held-out (via regra DT) — são as
    # amostras mais informativas para o teste de ablação, conforme o artigo:
    # "the effect was much stronger in samples where that concept was highly active"

    resultados_ablacao = []

    for c in conceitos_finais:
        k = c["fator"]
        dt_k = c["dt_model"]
        v_dir = construir_direcao_em_X(c, feature_names)

        if v_dir is None:
            c["destruction_delta"] = None
            c["sufficiency_delta"] = None
            continue

        # Amostras held-out onde o conceito está ativo (regra DT = 1)
        mask_ativo = dt_k.predict(X_held_orig) == 1
        if mask_ativo.sum() < 3:
            c["destruction_delta"] = None
            c["sufficiency_delta"] = None
            continue

        X_ativo = X_held_orig[mask_ativo]

        # Predição original (baseline) nessas amostras
        p_original = predizer_proba(modelo_tabpfn, X_ativo)

        # ── Destruction Test (Necessidade) — eq. 8 ──────────────────────────
        # Remove a componente na direção do conceito: x_destroyed = x - (x·v)v
        proj_escalar = X_ativo @ v_dir  # (n,)
        X_destruido = X_ativo - np.outer(proj_escalar, v_dir)
        p_destruido = predizer_proba(modelo_tabpfn, X_destruido)
        # Artigo (seção 5.6): "|Δdestroy| < 0.02 on average" — usa o valor
        # ABSOLUTO do delta por amostra antes de tirar a média. Isso evita que
        # deltas positivos e negativos se cancelem entre amostras diferentes.
        delta_por_amostra = np.abs(p_original - p_destruido)
        delta_destruction = float(np.mean(delta_por_amostra))

        # ── Sufficiency Test — eq. 9 ─────────────────────────────────────────
        # Mantém SOMENTE a componente na direção do conceito: x_suf = (x·v)v
        X_suficiente = np.outer(proj_escalar, v_dir)
        p_suficiente = predizer_proba(modelo_tabpfn, X_suficiente)

        # MÉTRICA CORRIGIDA: a razão direta p_suficiente/p_original explode
        # quando p_original é pequeno (ex: p_original=0.12, p_suficiente=0.26
        # → razão=2.16, fora do intervalo [0,1] e sem leitura como "fração
        # retida"). Em vez disso, medimos a PROXIMIDADE entre a predição
        # suficiente e a original por amostra: 1 − |p_suficiente − p_original|.
        # Fica sempre em [0,1]: 1.0 = predição suficiente idêntica à original,
        # 0.0 = predições opostas (uma em 0, outra em 1).
        proximidade_por_amostra = 1.0 - np.abs(p_suficiente - p_original)
        retencao_sufficiency = float(np.mean(proximidade_por_amostra))

        # Mantido apenas como diagnóstico complementar (não usado no filtro)
        delta_sufficiency = float(np.mean(p_suficiente) - np.mean(p_original))

        c["destruction_delta"] = round(delta_destruction, 4)
        c["sufficiency_delta"] = round(delta_sufficiency, 4)
        c["sufficiency_retencao"] = round(retencao_sufficiency, 4)
        c["n_amostras_ablacao"] = int(mask_ativo.sum())

        resultados_ablacao.append(
            {
                "fator": k,
                "n_amostras_ativas": int(mask_ativo.sum()),
                "p_original_medio": round(float(np.mean(p_original)), 4),
                "p_destruido_medio": round(float(np.mean(p_destruido)), 4),
                "destruction_delta": c["destruction_delta"],
                "p_suficiente_medio": round(float(np.mean(p_suficiente)), 4),
                "sufficiency_retencao": c["sufficiency_retencao"],
                "necessario": bool(
                    delta_destruction >= 0.02
                ),  # Δp médio ≥ 0.02 → necessário
                "suficiente": bool(
                    retencao_sufficiency >= 0.7
                ),  # proximidade ≥70% → suficiente
            }
        )

    df_ablacao = pd.DataFrame(resultados_ablacao).sort_values(
        "destruction_delta", key=lambda s: s.abs(), ascending=False
    )
    df_ablacao.to_csv(os.path.join(pasta_ml, "testes_ablacao.csv"), index=False)

    print("\n  Resultados dos testes de ablação:")
    print(df_ablacao.to_string(index=False))

    n_necessarios = int(df_ablacao["necessario"].sum())
    n_suficientes = int(df_ablacao["suficiente"].sum())
    print(f"\n  Conceitos necessários (|Δp|≥0.02): {n_necessarios} / {len(df_ablacao)}")
    print(
        f"  Conceitos suficientes (proximidade≥70%): {n_suficientes} / {len(df_ablacao)}"
    )

    # ── Plot: Destruction Δp vs Sufficiency proximidade ──────────────────────
    if len(df_ablacao) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(df_ablacao) * 0.4 + 2)))

        labels_abl = [f"F{f}" for f in df_ablacao["fator"]]
        cores_destr = ["#DC2626" if n else "#94A3B8" for n in df_ablacao["necessario"]]
        cores_suf = ["#16A34A" if s else "#94A3B8" for s in df_ablacao["suficiente"]]

        axes[0].barh(labels_abl, df_ablacao["destruction_delta"], color=cores_destr)
        axes[0].axvline(0, color="black", linewidth=0.5)
        axes[0].set_xlabel("Δp (predição original − destruída)")
        axes[0].set_title("Destruction Test\n(vermelho = necessário, |Δp|≥0.02)")

        axes[1].barh(labels_abl, df_ablacao["sufficiency_retencao"], color=cores_suf)
        axes[1].axvline(
            0.7, color="black", linestyle="--", linewidth=0.5, label="limiar 0.7"
        )
        axes[1].set_xlim(0, 1)
        axes[1].set_xlabel("Proximidade (1 − |p_suficiente − p_original|)")
        axes[1].set_title("Sufficiency Test\n(verde = suficiente, proximidade≥0.7)")
        axes[1].legend(fontsize=8)

        salvar_plot("testes_ablacao", pasta_ml)

    print(f"\n  Interpretação:")
    print(f"  - 'Necessário': remover a direção do conceito muda a predição em ≥2pp")
    print(f"  - 'Suficiente': usar SÓ essa direção reproduz a predição original")
    print(f"    com proximidade ≥70% (1 − |p_suficiente − p_original| ≥ 0.7)")
    print(f"  - Conceitos necessários E suficientes são os candidatos mais fortes a")
    print(f"    'drivers' causais da decisão do modelo, não apenas correlações.")

    df_final = pd.DataFrame(
        [
            {
                "fator": c["fator"],
                "tcav": c["tcav_mean"],
                "tcav_std": c["tcav_std"],
                "p_adj": c["p_adj"],
                "cohens_d": c["cohens_d"],
                "desvio_chance": c["desvio_chance"],
                "importancia_rf": round(c["importancia_rf"], 4),
                "precisao_dt": c["precisao"],
                "recall_dt": c["recall"],
                "regra": c["regra"],
                "top5_features": ", ".join(c["top5_features"]),
                "interpretacao": "RISCO" if c["tcav_mean"] > 0.5 else "PROTETOR",
                "destruction_delta": c.get("destruction_delta"),
                "sufficiency_retencao": c.get("sufficiency_retencao"),
            }
            for c in conceitos_finais
        ]
    ).sort_values("importancia_rf", ascending=False)

    df_final.to_csv(os.path.join(pasta_ml, "conceitos_interpretaveis.csv"), index=False)

    print("\n  Top conceitos:")
    print(
        df_final[
            [
                "fator",
                "tcav",
                "importancia_rf",
                "interpretacao",
                "regra",
                "top5_features",
            ]
        ].to_string(index=False)
    )

    # ── Plots ────────────────────────────────────────────────────────────────

    n_plot = len(df_final)
    altura = max(4, n_plot * 0.55 + 2)
    labels = [
        f"F{row['fator']}: {row['regra'][:40]}{'...' if len(row['regra']) > 40 else ''}"
        for _, row in df_final.iterrows()
    ]
    cores = [
        "#DC2626" if r == "RISCO" else "#16A34A" for r in df_final["interpretacao"]
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, altura))

    # TCAV
    axes[0].barh(labels, df_final["tcav"], color=cores, alpha=0.85)
    axes[0].axvline(
        0.5, color="black", linestyle="--", linewidth=1, label="Chance (0.5)"
    )
    axes[0].set_xlabel("TCAV Score")
    axes[0].set_title("TCAV Score\n(vermelho=risco, verde=protetor)")
    axes[0].set_xlim(0, 1)
    axes[0].legend(fontsize=8)

    # RF importance
    axes[1].barh(labels, df_final["importancia_rf"], color="#7C3AED", alpha=0.85)
    axes[1].set_xlabel("Importância RF")
    axes[1].set_title("Importância Random Forest\npor Conceito SAE")

    salvar_plot("conceitos_finais", pasta_ml)

    # Ativação média por classe (held-out)
    n_show = min(8, len(conceitos_finais))
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, c in enumerate(conceitos_finais[:n_show]):
        k = c["fator"]
        ativ = acts_held_out[:, k]
        for cls in np.unique(y_held_out_labels):
            mean_act = ativ[y_held_out_labels == cls].mean()
            ax.bar(
                i + int(cls) * 0.4,
                mean_act,
                width=0.35,
                label=f"Classe {cls}" if i == 0 else "",
                color=["#2563EB", "#DC2626"][int(cls)],
                alpha=0.8,
            )
    ax.set_xticks(range(n_show))
    ax.set_xticklabels([f"F{c['fator']}" for c in conceitos_finais[:n_show]])
    ax.set_xlabel("Conceito SAE")
    ax.set_ylabel("Ativação Média")
    ax.set_title("Ativação Média por Conceito e Classe (Held-Out)")
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles[:2], lbls[:2])
    salvar_plot("ativacao_por_classe", pasta_ml)

    # t-SNE dos embeddings colorido pelo top conceito
    if len(test_emb) <= 1000:
        top_fator = conceitos_finais[0]["fator"]
        ativ_top = acts_held_out[:, top_fator]
        tsne = TSNE(
            n_components=2,
            random_state=RANDOM_STATE,
            perplexity=min(30, len(emb_held_out) - 1),
        )
        emb_2d = tsne.fit_transform(emb_held_out)
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(
            emb_2d[:, 0], emb_2d[:, 1], c=ativ_top, cmap="RdBu_r", alpha=0.7, s=20
        )
        plt.colorbar(sc, label=f"Ativação Fator {top_fator}")
        plt.title(
            f"t-SNE — Fator {top_fator} (top conceito)\n{conceitos_finais[0]['regra'][:60]}"
        )
        salvar_plot(f"tsne_fator_{top_fator}", pasta_ml)

else:
    print("\n  Nenhum conceito interpretável encontrado.")
    print("  Parâmetros para relaxar:")
    print(f"    DT_PRECISION_MIN = {DT_PRECISION_MIN} → tente 0.80")
    print(f"    DT_RECALL_MIN    = {DT_RECALL_MIN} → tente 0.15")
    print(f"    TCAV_EFFECT_MIN  = {TCAV_EFFECT_MIN} → tente 0.05")
    print(f"    SAE_SPARSITY     = {SAE_SPARSITY} → tente 0.01")

# ── Salvar métricas do SAE ───────────────────────────────────────────────────
pd.DataFrame(
    [
        {
            "embedding_dim": EMBEDDING_DIM,
            "latent_dim": LATENT_DIM,
            "expansion_factor": EXPANSION_FACTOR,
            "near_zero_rate": round(near_zero_rate, 4),
            "active_per_sample": round(active_per_sample, 2),
            "dead_factors": len(dead_factors),
            "active_factors": len(active_factors),
            "high_sim_pairs": int(high_sim_pairs),
            "candidatos_dt": len(conceitos_candidatos),
            "conceitos_finais": len(conceitos_finais),
        }
    ]
).to_csv(os.path.join(pasta_ml, "metricas_sae.csv"), index=False)

print(f"\n✓ Pipeline concluído. Resultados em: {base_dir}/")
