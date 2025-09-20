# exercicio_DSteam
Este script implementa um pipeline completo de Machine Learning para prever risco de inadimplência (default), então, a ideia foi criar um modelo que conseguisse prever risco de inadimplência.
Ele foi criado para testes iniciais (baseline), permitindo:
1. Carregar e explorar rapidamente os dados.
2. Tratar valores ausentes e transformar variáveis numéricas e categóricas.
3. Treinar e comparar modelos (Logistic Regression e Random Forest).
4. Avaliar a performance com métricas de classificação (ROC AUC, etc.).
5. Salvar o melhor modelo em arquivo para uso posteriormente.
6. Fazer previsões em novos dados a partir de um dicionário (input_dict).

Saída principal:
- Arquivo `xhealth_best_pipeline.joblib` com o modelo final treinado.
- Relatório em `xhealth_model_report.json` com informações de dataset e performance.
