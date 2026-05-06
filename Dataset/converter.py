import os
import re
import numpy as np
import pandas as pd

# ==============================
# CONFIGURAÇÃO
# ==============================
PASTA_DATASET = "./Dataset/Raw_Data"
PASTA_SAIDA = "./Dataset/Processed_Data"
PASTA_PROCESSADA = "./Dataset/Processed_Data"
ARQUIVO_FINAL = os.path.join(PASTA_PROCESSADA, "dataset_unificado.csv")

NUM_LINHAS_POR_REGISTRO = 10
NUM_COLUNAS = 75  # sem "name"

colunas = [
    "id",
    "ccf",
    "age",
    "sex",
    "painloc",
    "painexer",
    "relrest",
    "pncaden",
    "cp",
    "trestbps",
    "htn",
    "chol",
    "smoke",
    "cigs",
    "years",
    "fbs",
    "dm",
    "famhist",
    "restecg",
    "ekgmo",
    "ekgday",
    "ekgyr",
    "dig",
    "prop",
    "nitr",
    "pro",
    "diuretic",
    "proto",
    "thaldur",
    "thaltime",
    "met",
    "thalach",
    "thalrest",
    "tpeakbps",
    "tpeakbpd",
    "dummy",
    "trestbpd",
    "exang",
    "xhypo",
    "oldpeak",
    "slope",
    "rldv5",
    "rldv5e",
    "ca",
    "restckm",
    "exerckm",
    "restef",
    "restwm",
    "exeref",
    "exerwm",
    "thal",
    "thalsev",
    "thalpul",
    "earlobe",
    "cmo",
    "cday",
    "cyr",
    "num",
    "lmt",
    "ladprox",
    "laddist",
    "diag",
    "cxmain",
    "ramus",
    "om1",
    "om2",
    "rcaprox",
    "rcadist",
    "lvx1",
    "lvx2",
    "lvx3",
    "lvx4",
    "lvf",
    "cathef",
    "junk",
]


def juntar_datasets():
    dfs = []

    for arquivo in os.listdir(PASTA_PROCESSADA):
        if arquivo.endswith("_convertido.csv"):
            caminho = os.path.join(PASTA_PROCESSADA, arquivo)

            print(f"Lendo: {arquivo}")

            df = pd.read_csv(caminho)

            # -----------------------------
            # Criar coluna de origem
            # -----------------------------
            nome_origem = arquivo.replace("_convertido.csv", "")
            df["dataset_origem"] = nome_origem

            dfs.append(df)

    df_final = pd.concat(dfs, ignore_index=True)

    df_final.to_csv(ARQUIVO_FINAL, index=False)

    print("\nDataset unificado salvo em:")
    print(ARQUIVO_FINAL)
    print("Shape final:", df_final.shape)


def converter_arquivo(caminho_entrada, caminho_saida):
    print(f"\nProcessando: {caminho_entrada}")

    with open(caminho_entrada, "r", encoding="latin1") as f:
        linhas = f.readlines()

    registros = []
    buffer = []

    for linha in linhas:
        buffer.append(linha.strip())

        if len(buffer) == NUM_LINHAS_POR_REGISTRO:
            texto = " ".join(buffer)

            # corrigir números grudados
            texto = re.sub(r"(\d)-(\d)", r"\1 -\2", texto)

            tokens = texto.split()

            valores = []
            for t in tokens:
                try:
                    valores.append(float(t))
                except:
                    continue  # ignora "name"

            if len(valores) == NUM_COLUNAS:
                registros.append(valores)
            else:
                print(f"Registro ignorado (tamanho {len(valores)})")

            buffer = []

    print(f"Registros válidos: {len(registros)}")

    if len(registros) == 0:
        print("Nenhum dado válido, arquivo ignorado.")
        return

    df = pd.DataFrame(registros, columns=colunas)

    # limpeza básica
    df = df.replace([-9, -9.0], np.nan)

    df.to_csv(caminho_saida, index=False)

    print(f"Salvo em: {caminho_saida}")


# ==============================
# PROCESSAR TODOS OS .data
# ==============================

os.makedirs(PASTA_SAIDA, exist_ok=True)
for arquivo in os.listdir(PASTA_DATASET):
    if arquivo.endswith(".data"):
        caminho_entrada = os.path.join(PASTA_DATASET, arquivo)

        nome_saida = arquivo.replace(".data", "_convertido.csv")
        caminho_saida = os.path.join(PASTA_SAIDA, nome_saida)

        converter_arquivo(caminho_entrada, caminho_saida)

juntar_datasets()

print("\nConversão finalizada para todos os arquivos!")
