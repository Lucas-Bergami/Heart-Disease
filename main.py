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
RANDOM_STATE = 42
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
# BLOCO SAE + TCAV
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


# ── 6. DECISION TREE POR FATOR ───────────────────────────────────────────────
# Threshold = percentil 50 das ativações positivas (artigo seção 3.5)
# Retém fator somente se precisão ≥ 0.9 E recall ≥ 0.25

print(
    f"\n[4/6] Decision Trees por fator SAE (P≥{DT_PRECISION_MIN}, R≥{DT_RECALL_MIN})..."
)

conceitos_candidatos = []

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
        regra = export_text(dt, feature_names=feature_names, max_depth=2)
        regra_resumida = next(
            (
                l.strip().replace("|--- ", "")
                for l in regra.split("\n")
                if l.strip() and "|---" in l
            ),
            "ver DT",
        )[:80]

        conceitos_candidatos.append(
            {
                "fator": k,
                "precisao": round(precisao, 3),
                "recall": round(recall, 3),
                "threshold_k": threshold_k,
                "regra": regra_resumida,
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

    resultados_tcav.append(
        {
            "fator": k,
            "precisao": conceito["precisao"],
            "recall": conceito["recall"],
            "regra": conceito["regra"],
            "tcav_mean": round(tcav_mean, 4),
            "tcav_std": round(tcav_std, 4),
            "p_valor": p_val,
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

    # Filtro final: significância + efeito mínimo
    conceitos_finais = [
        r
        for r in resultados_tcav
        if r["p_adj"] < FDR_ALPHA and r["desvio_chance"] >= TCAV_EFFECT_MIN
    ]

print(
    f"  Aprovados (p_adj<{FDR_ALPHA}, |TCAV-0.5|≥{TCAV_EFFECT_MIN}): {len(conceitos_finais)}"
)

if not conceitos_finais:
    print("\n  ⚠ Nenhum conceito passou o filtro TCAV.")
    print("  Tente: reduzir TCAV_EFFECT_MIN para 0.05, ou DT_RECALL_MIN para 0.15")


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

    # ── DataFrame de resultados ──────────────────────────────────────────────
    df_final = pd.DataFrame(
        [
            {
                "fator": c["fator"],
                "tcav": c["tcav_mean"],
                "tcav_std": c["tcav_std"],
                "p_adj": c["p_adj"],
                "desvio_chance": c["desvio_chance"],
                "importancia_rf": round(c["importancia_rf"], 4),
                "precisao_dt": c["precisao"],
                "recall_dt": c["recall"],
                "regra": c["regra"],
                "top5_features": ", ".join(c["top5_features"]),
                "interpretacao": "RISCO" if c["tcav_mean"] > 0.5 else "PROTETOR",
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
