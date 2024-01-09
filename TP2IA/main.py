import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Carregar dados
dados_ml_path = '/Users/tomassilva/Documents/TP2IA/Dados_ML.xlsx'
dados_ml = pd.read_excel(dados_ml_path)

# Dividir dados
X = dados_ml.drop('Abandono', axis=1)
y = dados_ml['Abandono']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalização dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Função para treinar e avaliar modelos
def train_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Model: {model.__class__.__name__}")
    print("Cross-Validation Scores:", scores)
    print("Average Score:", scores.mean())
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # Calculando a curva ROC e AUC
    y_scores = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Regressão Logística
log_reg = LogisticRegression(max_iter=1000)
train_evaluate_model(log_reg, X_train_scaled, y_train, X_test_scaled, y_test)

# Árvore de Decisão
decision_tree = DecisionTreeClassifier()
train_evaluate_model(decision_tree, X_train, y_train, X_test, y_test)

# Rede Neural (Perceptron Multicamadas)
mlp = MLPClassifier(max_iter=1000)
train_evaluate_model(mlp, X_train_scaled, y_train, X_test_scaled, y_test)
