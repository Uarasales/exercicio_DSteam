# baseline_pipeline_xhealth.py
# Script para:
# - Explorar rapidamente os dados (EDA)
# - Pré-processar (tratamento de nulos, normalização, encoding)
# - Treinar dois modelos baseline (LogisticRegression e RandomForest)
# - Comparar via ROC AUC
# - Salvar o melhor modelo
# - Implementar função predict(input_dict) no formato exigido


# Importação de bibliotecas

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings("ignore")  # ignora avisos para não poluir a saída

# Carregamento dos dados
CSV_PATH = "dataset_2021-5-26-10-14.csv"  

# Verifica se o arquivo existe; caso contrário, avisa o usuário
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Arquivo não encontrado em {CSV_PATH}. Faça upload e rode novamente.")

# Lê o CSV (separado por tabulação) e troca valores "missing" por NaN
df = pd.read_csv(CSV_PATH, sep="\t", encoding="utf-8")
df = df.replace("missing", np.nan)

print("Dimensão do dataset:", df.shape)      # linhas x colunas
print("\nColunas:\n", df.columns.tolist())   # nomes das colunas

# EDA rápida (exploração inicial) 
print("\nValores nulos por coluna (top 20):")
print(df.isna().sum().sort_values(ascending=False).head(20))

# Verifica a distribuição do target (coluna 'default')
print("\nDistribuição do target 'default':")
print(df['default'].value_counts(dropna=False))

# Taxa de default média (proporção de clientes com default=1)
print("\nROC-friendly stats - proporção de defaults:", df['default'].mean())

# Definir features e target
target_col = 'default'  

# Colunas que podem ser removidas do treino (exemplo: IDs, variáveis de tempo)
drop_if_exist = ['default_3months']  

# Features = todas as colunas exceto o target
feature_cols = [c for c in df.columns if c != target_col]

X = df[feature_cols].copy()         # variáveis de entrada
y = df[target_col].astype(int).copy()  # variável alvo (0 = não deu default, 1 = deu)

# Identifica automaticamente tipos de variáveis
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
print("\nNuméricas:", num_cols)
print("Categóricas:", cat_cols)

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.20, random_state=42
)
# stratify=y → garante que a proporção de classes (0/1) seja mantida
print("\nSplit feito. Treino:", X_train.shape, "Teste:", X_test.shape)

# Pré-processamento

# Para variáveis numéricas: preenche nulos com a mediana + normaliza
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Para variáveis categóricas: preenche nulos com "MISSING" + one-hot encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combina numéricas e categóricas em um único pré-processador
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
], remainder='drop')

# Pipelines dos modelos

# Pipeline 1: Regressão Logística
pipe_lr = Pipeline(steps=[
    ('preproc', preprocessor),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=2000, solver='liblinear'))
])

# Pipeline 2: Random Forest
pipe_rf = Pipeline(steps=[
    ('preproc', preprocessor),
    ('clf', RandomForestClassifier(n_jobs=-1, class_weight='balanced', random_state=42))
])

# Validação cruzada + busca de hiperparâmetros

cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

# Grid Search - Logistic Regression
lr_params = {
    'clf__C': [0.01, 0.1, 1.0],
    'clf__penalty': ['l2']
}
gs_lr = GridSearchCV(pipe_lr, lr_params, cv=cv, scoring='roc_auc', n_jobs=1)
gs_lr.fit(X_train, y_train)
print("\nBest LR params:", gs_lr.best_params_)
lr_best = gs_lr.best_estimator_
lr_proba = lr_best.predict_proba(X_test)[:,1]
print("LR Test ROC AUC:", roc_auc_score(y_test, lr_proba))

# Grid Search - Random Forest
rf_params = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [6, 12, None],
    'clf__min_samples_split': [2, 10]
}
gs_rf = GridSearchCV(pipe_rf, rf_params, cv=cv, scoring='roc_auc', n_jobs=1)
gs_rf.fit(X_train, y_train)
print("\nBest RF params:", gs_rf.best_params_)
rf_best = gs_rf.best_estimator_
rf_proba = rf_best.predict_proba(X_test)[:,1]
print("RF Test ROC AUC:", roc_auc_score(y_test, rf_proba))

# Escolher melhor modelo e avaliar

models = {
    'logistic': (lr_best, roc_auc_score(y_test, lr_proba)),
    'random_forest': (rf_best, roc_auc_score(y_test, rf_proba))
}
# Escolhe o modelo com maior ROC AUC
best_name = max(models.items(), key=lambda x: x[1][1])[0]
best_model = models[best_name][0]
best_score = models[best_name][1]
print(f"\nMelhor modelo: {best_name} com ROC AUC = {best_score:.4f}")

# Avaliação com threshold fixo (0.5)
best_proba = best_model.predict_proba(X_test)[:,1]
best_pred = (best_proba >= 0.5).astype(int)
print("\nClassification report (threshold 0.5):\n", classification_report(y_test, best_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, best_pred))

# PR AUC (precision-recall area under curve)
precision, recall, _ = precision_recall_curve(y_test, best_proba)
pr_auc = auc(recall, precision)
print(f"PR AUC: {pr_auc:.4f}")

# Importância das features (quando possível) 

if best_name == 'random_forest':
    # Extrai nomes de features após o one-hot
    preproc = best_model.named_steps['preproc']
    num_features = num_cols
    cat_features = []
    if len(cat_cols) > 0:
        ohe = preproc.named_transformers_['cat'].named_steps['onehot']
        cat_names = ohe.get_feature_names_out(cat_cols).tolist()
        cat_features = cat_names
    all_features = num_features + cat_features
    importances = best_model.named_steps['clf'].feature_importances_
    feat_imp = pd.Series(importances, index=all_features).sort_values(ascending=False).head(30)
    print("\nTop 30 feature importances (RF):\n", feat_imp)
else:
    # Para regressão logística seria preciso mapear coeficientes (mais trabalhoso com OHE)
    print("\nModelo logistic escolhido — coeficientes disponíveis se precisar.")

# Salvar modelo final

MODEL_PATH = "xhealth_best_pipeline.joblib"
joblib.dump(best_model, MODEL_PATH)
print("\nModelo salvo em:", MODEL_PATH)

# Função de predição 

def predict_from_dict(input_dict, model_path=MODEL_PATH, feature_columns=None):
    """
    Função de predição para uso em produção.
    input_dict: dicionário com as variáveis de entrada
    Retorna: {'default': 0/1, 'probability_default': float}
    """
    model = joblib.load(model_path)
    
    # Se não passar as colunas, pega as usadas no treino
    if feature_columns is None:
        feature_columns = X.columns.tolist()
    
    # Garante que todas as features estão no formato certo
    row = {c: input_dict.get(c, np.nan) for c in feature_columns}
    X_row = pd.DataFrame([row], columns=feature_columns)
    
    # Faz predição
    proba = model.predict_proba(X_row)[:,1][0]
    pred = int(proba >= 0.5)
    return {"default": pred, "probability_default": float(proba)}

# Exemplo de uso (valores fictícios):
example = {c: None for c in X.columns.tolist()}
print("\nExemplo de predição com valores vazios:", predict_from_dict(example))

# Relatório final

report = {
    'n_rows': df.shape[0],             # nº de registros
    'n_features': len(feature_cols),   # nº de variáveis
    'target_mean': float(y.mean()),    # média do target (taxa default)
    'best_model': best_name,           # modelo escolhido
    'best_roc_auc': float(best_score)  # métrica de performance
}
print("\nResumo:", report)

# Salva relatório em JSON
pd.Series(report).to_json("xhealth_model_report.json")
print("Relatório salvo em xhealth_model_report.json")
