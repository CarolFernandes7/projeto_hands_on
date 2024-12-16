import pandas as pd

def load_raw_data(filepath):
    """Carrega os dados brutos."""
    print(f"Carregando dados de: {filepath}")
    return pd.read_csv(filepath)

def clean_data(df):
    """Limpa e prepara os dados."""
    print("Colunas originais:", df.columns)
    
    # Excluir colunas irrelevantes
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    
    # Criar variáveis dummies para colunas categóricas
    df = pd.get_dummies(df, columns=['Gender', 'Geography', 'Card Type'], drop_first=True)
    
    print("Colunas após limpeza:", df.columns)
    return df

def save_to_staging(df, output_path):
    """Salva os dados tratados na camada staging."""
    df.to_csv(output_path, index=False)
    print(f"Dados tratados salvos em: {output_path}")

if __name__ == "__main__":
    raw_path = 'data/raw/Bank_Customer_Churn_Prediction.csv'
    staging_path = 'data/staging/staging_churn.csv'

    print("Iniciando pipeline ETL...")
    try:
        df_raw = load_raw_data(raw_path)
        print("Dados carregados com sucesso!")
        print(df_raw.head())  # Mostra as primeiras linhas

        df_cleaned = clean_data(df_raw)
        print("Dados limpos com sucesso!")

        save_to_staging(df_cleaned, staging_path)
        print("Pipeline concluído com sucesso!")
    except Exception as e:
        print("Erro no pipeline ETL:", e)