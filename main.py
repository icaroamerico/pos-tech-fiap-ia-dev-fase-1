import pandas as pd

data = {"a": [1], "b": [1]}
df = pd.DataFrame(data)
resultado = df["a"] + df["b"]
print(f"Resultado de 1 + 1 = {resultado[0]}")
