# Projeto Hands On - Mackenzie Eng Dados 


Documentação do Script de Treinamento

O que o Script Faz?

O script train_model.py realiza as seguintes etapas:

	1.	Carrega e limpa os dados:
	•	Carrega o arquivo staging_churn.csv da camada staging.
	•	Garante que todas as colunas são numéricas para o modelo.
 
	2.	Aplica SMOTE:
	•	Balanceia as classes para evitar problemas com dados desbalanceados.
 
	3.	Divide os dados:
	•	Separa os dados em treino (70%), validação (15%) e teste (15%).
 
	4.	Treina o RandomForest:
	•	Treina o modelo com os seguintes hiperparâmetros ajustados:
	•	200 árvores, max_depth=10, min_samples_split=5, min_samples_leaf=2.
 
	5.	Avalia o modelo:
	•	Avaliação detalhada nos conjuntos de validação e teste.
	•	Realiza validação cruzada para garantir a robustez do modelo.
 
	6.	Visualiza os resultados:
	•	Importância das Variáveis: Gráfico mostrando o impacto de cada variável no modelo.
	•	Matriz de Confusão: Análise detalhada dos resultados de previsão.
 
	7.	Compara com outros modelos:
	•	Logistic Regression.
	•	XGBoost.
 
	8.	Interpreta o modelo com SHAP:
	•	Utiliza SHAP (SHapley Additive exPlanations) para explicar as previsões do modelo.
	•	Gera um gráfico SHAP summary plot.
 
	9.	Salva o modelo:
	•	O modelo RandomForest treinado é salvo em:
	•	models/random_forest_churn.pkl.

Como Rodar o Script
	
 1.	Certifique-se de ter instalado as bibliotecas necessárias:

 ``pip install pandas scikit-learn imbalanced-learn matplotlib xgboost shap joblib``

 
	2.	Execute o script:

``python3 scripts/train_model.py ``

Saída Esperada

	1.	Métricas detalhadas:
	•	Acurácia, Precision, Recall, F1-Score para:
	•	RandomForest.
	•	Logistic Regression.
	•	XGBoost.
 
	2.	Validação Cruzada:
	•	Acurácia média obtida através de 5 folds.
 
	3.	Gráficos:
	•	Importância das Variáveis no modelo RandomForest.
	•	Matriz de Confusão para os resultados de teste.
	•	SHAP summary plot para análise interpretativa do modelo.
 
	4.	Modelo salvo:
	•	O modelo treinado é salvo em:
	•	models/random_forest_churn.pkl.
 
 Exemplo de Gráficos Gerados

Importância das Variáveis


Matriz de Confusão

SHAP Summary Plot


Estrutura do Projeto
``projeto_hands_on/
│
├── data/                     # Dados usados no projeto
│   ├── raw/                  # Dados brutos
│   ├── staging/              # Dados tratados
│   └── analytics/            # Dados finais
│
├── scripts/                  # Scripts Python
│   ├── etl_pipeline.py       # Pipeline ETL
│   └── train_model.py        # Treinamento e avaliação do modelo
│
├── models/                   # Modelos treinados
│   └── random_forest_churn.pkl
│
├── notebooks/                # Jupyter Notebooks
│   └── eda_churn.ipynb       # Análise exploratória dos dados
│
├── terraform/                # Scripts de IaC (Terraform)
│   └── main.tf               # Provisionamento da infraestrutura
│
├── README.md                 # Documentação do projeto
└── requirements.txt          # Dependências do projeto
``
