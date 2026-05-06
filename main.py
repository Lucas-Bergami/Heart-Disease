import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime

# ==============================
# 2. CONFIGURAÇÃO
# ==============================
caminho_csv = "./Dataset/Processed_Data/dataset_unificado.csv"

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


def salvar_plot(nome, pasta):
    caminho = os.path.join(pasta, f"{nome}.png")
    plt.savefig(caminho, bbox_inches="tight")
    plt.close()


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

# ==============================
# 3. CARREGAR DADOS
# ==============================
print(f"Carregando: {caminho_csv}")
df = pd.read_csv(caminho_csv)

print("Dataset carregado")
print(df.shape)

# ==============================
# 4. LIMPEZA
# ==============================

coluna_origem = "dataset_origem" if "dataset_origem" in df.columns else None

# ------------------------------
# 1. Converter para numérico
# ------------------------------
for col in df.columns:
    if col != coluna_origem:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ------------------------------
# 2. Remover colunas vazias
# ------------------------------
colunas_vazias = [col for col in df.columns if df[col].isna().all()]
df = df.drop(columns=colunas_vazias)

print("Colunas removidas:", colunas_vazias)

# ------------------------------
# 3. Remover colunas indesejadas
# ------------------------------
df = df.drop(columns=[c for c in colunas_remover if c in df.columns])

# ------------------------------
# 4. DEFINIR COLUNAS
# ------------------------------
colunas_numericas = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

if "num" in colunas_numericas:
    colunas_numericas.remove("num")

# categóricas = poucas categorias
colunas_categoricas = [col for col in colunas_numericas if df[col].nunique() < 10]

# separar de verdade
colunas_numericas = [col for col in colunas_numericas if col not in colunas_categoricas]

# ------------------------------
# 5. PREENCHIMENTO
# ------------------------------
df[colunas_numericas] = df[colunas_numericas].fillna(df[colunas_numericas].median())

for col in colunas_categoricas:
    moda = df[col].mode()
    if not moda.empty:
        df[col] = df[col].fillna(moda[0])

# ------------------------------
# 6. TARGET BINÁRIO
# ------------------------------
if "num" in df.columns:
    df["num"] = (df["num"] > 0).astype(int)

print("Shape após limpeza:", df.shape)
# ==============================
# 6. INFORMAÇÕES
# ==============================
print("\n=== INFO ===")
print(df.info())

print("\n=== DESCRITIVO ===")
print(df.describe())

# ==============================
# GERAR PDF DE DESCRIÇÃO DO DATASET
# ==============================


def gerar_pdf_descricao(caminho_saida):
    if os.path.exists(caminho_saida):
        print("PDF já existe, pulando geração...")
        return

    print("Gerando PDF de descrição...")

    styles = getSampleStyleSheet()
    elementos = []

    def add_titulo(texto):
        elementos.append(Paragraph(f"<b>{texto}</b>", styles["Heading2"]))
        elementos.append(Spacer(1, 10))

    def add_texto(texto):
        elementos.append(Paragraph(texto, styles["BodyText"]))
        elementos.append(Spacer(1, 8))

    # ==============================
    # DESCRIÇÕES SIMPLIFICADAS
    # ==============================
    descricoes = {
        "age": "Idade do paciente em anos. Idades maiores aumentam o risco de doenças cardíacas.",
        "sex": "Sexo do paciente. 1 = masculino, 0 = feminino.",
        "painloc": "Local da dor no peito. 1 = região central do peito, 0 = outro local.",
        "painexer": "Dor provocada por esforço físico. 1 = sim, 0 = não.",
        "relrest": "Dor aliviada com repouso. 1 = sim, 0 = não.",
        "cp": "Tipo de dor no peito: "
        "1 = angina típica (fortemente relacionada ao coração), "
        "2 = angina atípica, "
        "3 = dor não cardíaca, "
        "4 = assintomático (sem dor).",
        "trestbps": "Pressão arterial em repouso (mmHg). Valores altos indicam hipertensão.",
        "htn": "Hipertensão. 1 = paciente tem pressão alta, 0 = não.",
        "chol": "Colesterol no sangue (mg/dl). Valores altos aumentam risco cardíaco.",
        "smoke": "Se o paciente fuma. 1 = sim, 0 = não.",
        "cigs": "Quantidade de cigarros fumados por dia.",
        "years": "Quantidade de anos como fumante.",
        "fbs": "Açúcar no sangue em jejum > 120 mg/dl. 1 = sim, 0 = não.",
        "dm": "Histórico de diabetes. 1 = sim, 0 = não.",
        "famhist": "Histórico familiar de doença cardíaca. 1 = sim, 0 = não.",
        "restecg": "Resultado do eletrocardiograma em repouso: "
        "0 = normal, "
        "1 = alteração leve, "
        "2 = possível problema mais sério no coração.",
        "dig": "Uso de digitalis durante exame. 1 = sim, 0 = não.",
        "prop": "Uso de beta-bloqueador durante exame. 1 = sim, 0 = não.",
        "nitr": "Uso de nitrato durante exame. 1 = sim, 0 = não.",
        "pro": "Uso de bloqueador de canal de cálcio. 1 = sim, 0 = não.",
        "diuretic": "Uso de diurético. 1 = sim, 0 = não.",
        "proto": "Protocolo do teste de esforço (tipo de exercício realizado). Diferentes números representam diferentes métodos de teste.",
        "thaldur": "Duração do teste de esforço (em minutos).",
        "thaltime": "Momento em que alteração cardíaca foi detectada durante o teste.",
        "met": "Capacidade física durante o exercício. Valores maiores indicam melhor condicionamento.",
        "thalach": "Frequência cardíaca máxima atingida. Valores baixos podem indicar problema cardíaco.",
        "thalrest": "Frequência cardíaca em repouso.",
        "tpeakbps": "Pressão arterial máxima durante exercício (parte sistólica).",
        "tpeakbpd": "Pressão arterial máxima durante exercício (parte diastólica).",
        "trestbpd": "Pressão arterial diastólica em repouso.",
        "exang": "Angina induzida por exercício. 1 = sim, 0 = não.",
        "xhypo": "Presença de condição específica detectada no exame. 1 = sim, 0 = não.",
        "oldpeak": "Alteração no exame (depressão ST). Valores maiores indicam maior risco cardíaco.",
        "slope": "Inclinação do segmento ST: "
        "1 = normal (ascendente), "
        "2 = plano (suspeito), "
        "3 = descendente (alto risco).",
        "rldv5": "Medição elétrica do coração em repouso (derivação V5).",
        "rldv5e": "Medição elétrica do coração durante esforço.",
        "ca": "Número de vasos sanguíneos principais afetados (0 a 3). Quanto maior, pior.",
        "restef": "Fração de ejeção do coração em repouso (capacidade de bombeamento).",
        "restwm": "Movimento da parede do coração em repouso: "
        "0 = normal, "
        "1 = leve alteração, "
        "2 = moderada, "
        "3 = grave.",
        "exeref": "Fração de ejeção durante exercício.",
        "exerwm": "Movimento da parede do coração durante exercício.",
        "thal": "Resultado de exame de perfusão: "
        "3 = normal, "
        "6 = defeito fixo (dano permanente), "
        "7 = defeito reversível (problema de fluxo sanguíneo).",
        "num": "Diagnóstico final: 0 = sem doença cardíaca, 1 = com doença cardíaca.",
        "lmt": "Artéria coronária esquerda principal. Indica presença de obstrução.",
        "ladprox": "Parte proximal da artéria descendente anterior esquerda.",
        "laddist": "Parte distal da artéria descendente anterior esquerda.",
        "diag": "Artéria diagonal (ramo da coronária).",
        "cxmain": "Artéria circunflexa principal.",
        "ramus": "Ramo intermediário da coronária.",
        "om1": "Primeiro ramo marginal obtuso.",
        "om2": "Segundo ramo marginal obtuso.",
        "rcaprox": "Parte proximal da artéria coronária direita.",
        "rcadist": "Parte distal da artéria coronária direita.",
        "lvf": "Função do ventrículo esquerdo (bombeamento do coração).",
        "dataset_origem": "Indica de qual base de dados o registro veio (ex: Cleveland, Hungary, etc).",
    }

    # ==============================
    # CONSTRUÇÃO DO PDF
    # ==============================
    add_titulo("Descrição do Dataset de Doença Cardíaca")

    add_texto(
        "Este documento explica, de forma simples, o significado de cada variável utilizada no dataset. "
        "O objetivo é facilitar a compreensão mesmo para quem não possui conhecimento médico."
    )

    for col, desc in descricoes.items():
        add_titulo(col)
        add_texto(desc)

    add_titulo("Observações Importantes")
    add_texto("- Algumas colunas foram removidas durante a limpeza.")

    # ==============================
    # GERAR PDF
    # ==============================
    doc = SimpleDocTemplate(caminho_saida)
    doc.build(elementos)

    print(f"PDF salvo em: {caminho_saida}")


# ==============================
# CRIAR PASTA DESCRIÇÃO
# ==============================
pasta_descricao = os.path.join(base_dir, "descricao")
os.makedirs(pasta_descricao, exist_ok=True)

caminho_pdf = os.path.join(pasta_descricao, "descricao_dataset.pdf")

gerar_pdf_descricao(caminho_pdf)

# ==============================
# 7. HISTOGRAMAS
# ==============================
df[colunas_numericas].hist(figsize=(18, 14))
plt.suptitle("Distribuição das Variáveis Numéricas")
salvar_plot("histogramas", pastas["histogramas"])

# ==============================
# 8. TARGET
# ==============================
plt.figure(figsize=(6, 4))
sns.countplot(x="num", data=df)
plt.title("Distribuição da Doença")
salvar_plot("target", pastas["categoricos"])

# ==============================
# 9. POR ORIGEM
# ==============================
if coluna_origem:
    plt.figure(figsize=(8, 5))
    sns.countplot(x="dataset_origem", hue="num", data=df)
    plt.title("Doença por Dataset")
    salvar_plot("doenca_por_origem", pastas["categoricos"])

# ==============================
# 10. BOXPLOTS
# ==============================
for col in colunas_numericas:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="num", y=col, data=df)
    plt.title(f"{col} vs Doença")
    salvar_plot(f"{col}_boxplot", pastas["comparacoes"])

# ==============================
# 11. COUNT PLOTS
# ==============================
for col in colunas_categoricas:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, hue="num", data=df)
    plt.title(f"{col} vs Doença")
    salvar_plot(f"{col}_count", pastas["categoricos"])

# ==============================
# 12. CORRELAÇÃO
# ==============================

df_numerico = df.select_dtypes(include=["int64", "float64"])

corr = df_numerico.corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Mapa de Correlação (somente numéricas)")
salvar_plot("correlacao_completa", pastas["correlacao"])

# ==============================
# 13. TOP CORRELAÇÕES
# ==============================
if "num" in df.columns:
    corr_target = corr["num"].sort_values(ascending=False)

    plt.figure(figsize=(8, 6))
    corr_target.drop("num").head(15).plot(kind="bar")
    plt.title("Top variáveis correlacionadas com doença")
    salvar_plot("top_corr", pastas["correlacao"])

print(f"\nResultados salvos em: {base_dir}")
