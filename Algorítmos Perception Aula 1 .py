# Importar bibliotecas necessárias
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Gerar dados de exemplo
X = np.array([[2, 3], [1, 5], [2, 1], [3, 6], [5, 7], [4, 2]])
y = np.array([0, 0, 0, 1, 1, 1])

# Dividir dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#O que é o Perceptron? Modelo de rede neural Artificial 
# Treinar o Perceptron
perceptron = Perceptron(max_iter=1000, tol=1e-3)
perceptron.fit(X_train, y_train)

# Fazer previsões e avaliar o modelo
y_pred = perceptron.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')