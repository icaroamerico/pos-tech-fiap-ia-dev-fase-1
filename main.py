import pandas as pd
import kagglehub
import os

print("Iniciando o download do dataset...")

path = kagglehub.dataset_download("prasoonkottarathil/polycystic-ovary-syndrome-pcos")

print(f"Download concluído! Arquivos salvos em: {path}")

arquivos = os.listdir(path)
print(f"Arquivos encontrados: {arquivos}")

arquivos_csv = [f for f in arquivos if f.endswith('.csv')]

if arquivos_csv:
    caminho_csv = os.path.join(path, arquivos_csv[0])
    df = pd.read_csv(caminho_csv)
    
    print("\nLeitura feita com sucesso! Aqui estão as 5 primeiras linhas:")
    print(df.head())
else:
    print("\nNenhum arquivo CSV encontrado no dataset.")