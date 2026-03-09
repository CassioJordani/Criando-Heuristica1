import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.datasets import load_digits
digits = load_digits()

# create Data Frame from digits data
df = pd.DataFrame(digits.data)

# Append target collum to the df
df['target'] = digits.target

# Prever de 0 e 9 para cada linha
df['predicao'] = [random.randint(0, 9) for _ in range(len(df))]

# Comparar se o target é igual a predição de forma aleatória e calcular acurácia
acertos = (df['target'] == df['predicao']).sum()
acuracia = acertos / len(df)

print(f'Total: {len(df)}')
print(f'Total de acertos: {acertos}')
print(f'Acurácia da heurística: {acuracia:.4f}')

# Mostrar uma imagem aleatória e a previsão
idx = random.randint(0, len(digits.images) - 1)
plt.gray()
plt.matshow(digits.images[idx])
plt.title(f"Real: {digits.target[idx]} | Predição: {df['predicao'].iloc[idx]}")
plt.show()
