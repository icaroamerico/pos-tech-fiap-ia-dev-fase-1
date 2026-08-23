from pathlib import Path
import warnings

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


class PCOSDataAnalysis:
    """Pipeline de classificacao de pacientes com sindrome dos ovarios policisticos."""

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.dt_model = None
        self.lr_model = None

    def carregar_dados(self):
        print("-" * 45)
        print("CARREGANDO DADOS DE PCOS")
        print("-" * 45)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Arquivo nao encontrado: {self.filepath}")
        self.df = pd.read_csv(self.filepath)
        target_column = "PCOS (Y/N)"
        if target_column not in self.df.columns:
            raise ValueError(f"A coluna obrigatoria '{target_column}' nao existe.")
        print("\nDados carregados com sucesso!")
        print(f"  - Dimensoes: {self.df.shape[0]} linhas x {self.df.shape[1]} colunas")
        print(f"  - Valores faltantes: {int(self.df.isna().sum().sum())}")
        print(f"  - Distribuicao do alvo: {self.df[target_column].value_counts().to_dict()}")
        return self.df

    def explorar_dados(self):
        print("\n" + "-" * 45)
        print("EXPLORACAO DOS DADOS")
        print("-" * 45)
        print("\n[METRICAS] Tipos de dados:")
        print(self.df.dtypes.value_counts())
        print("\nEstatisticas descritivas:")
        print(self.df.describe().transpose().head(10))

    def preprocessar_dados(self):
        print("\n" + "-" * 45)
        print("PRE-PROCESSAMENTO DOS DADOS")
        print("-" * 45)
        target_column = "PCOS (Y/N)"
        identifier_columns = ["Sl. No", "Patient File No."]
        feature_columns = [column for column in self.df.columns if column not in [target_column, *identifier_columns]]
        X = self.df[feature_columns].apply(pd.to_numeric, errors="coerce")
        y = pd.to_numeric(self.df[target_column], errors="coerce")
        valid_rows = y.notna()
        X = X.loc[valid_rows]
        y = y.loc[valid_rows].astype(int)
        print(f"\n Alvo: {target_column}")
        print(f"  - Identificadores removidos: {identifier_columns}")
        print(f"  - Features utilizadas: {X.shape[1]}")
        print(f"  - Valores ausentes nas features: {int(X.isna().sum().sum())}")
        print(f"  - Distribuicao do alvo: {y.value_counts().to_dict()}")
        return X, y

    def dividir_dados(self, X, y, test_size=0.2, random_state=42):
        print("\n" + "-" * 45)
        print("DIVIDINDO DADOS EM TREINO E TESTE")
        print("-" * 45)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print("\nDados divididos com sucesso!")
        print(f"  - Treinamento: {self.X_train.shape[0]} amostras")
        print(f"  - Teste: {self.X_test.shape[0]} amostras")

    def treinar_arvore_decisao(self, max_depth=10, random_state=42):
        print("\n" + "-" * 45)
        print("TREINANDO MODELO ARVORE DE DECISAO")
        print("-" * 45)
        self.dt_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)),
        ])
        self.dt_model.fit(self.X_train, self.y_train)
        tree = self.dt_model.named_steps["model"]
        print("\nModelo treinado com sucesso!")
        print(f"  - Profundidade final: {tree.get_depth()}")
        print(f"  - Numero de folhas: {tree.get_n_leaves()}")
        return self.dt_model

    def treinar_regressao_logistica(self, max_iter=1000, random_state=42):
        print("\n" + "-" * 45)
        print("TREINANDO MODELO: REGRESSAO LOGISTICA")
        print("-" * 45)
        self.lr_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=max_iter, random_state=random_state)),
        ])
        self.lr_model.fit(self.X_train, self.y_train)
        print("\nModelo treinado com sucesso!")
        return self.lr_model

    def avaliar_modelo(self, modelo, nome_modelo):
        print("\n" + "-" * 45)
        print(f"AVALIACAO: {nome_modelo.upper()}")
        print("-" * 45)
        y_pred_train = modelo.predict(self.X_train)
        y_pred_test = modelo.predict(self.X_test)
        accuracy_train, accuracy_test = self.calcular_acuracia(
            modelo, nome_modelo, y_pred_train, y_pred_test
        )
        metrics = {
            "accuracy_train": accuracy_train,
            "accuracy_test": accuracy_test,
            "recall_train": recall_score(self.y_train, y_pred_train, zero_division=0),
            "recall_test": recall_score(self.y_test, y_pred_test, zero_division=0),
            "f1_train": f1_score(self.y_train, y_pred_train, zero_division=0),
            "f1_test": f1_score(self.y_test, y_pred_test, zero_division=0),
        }
        print(f"\n{'CONJUNTO':<20} {'ACCURACY':<15} {'RECALL':<15} {'F1-SCORE':<15}")
        print("-" * 65)
        print(f"{'Treinamento':<20} {metrics['accuracy_train']:<15.4f} {metrics['recall_train']:<15.4f} {metrics['f1_train']:<15.4f}")
        print(f"{'Teste':<20} {metrics['accuracy_test']:<15.4f} {metrics['recall_test']:<15.4f} {metrics['f1_test']:<15.4f}")
        print("\n Matriz de confusao - teste:")
        print(confusion_matrix(self.y_test, y_pred_test))
        print("\nRelatorio de classificacao - teste:")
        print(classification_report(self.y_test, y_pred_test, target_names=["Sem PCOS", "Com PCOS"], zero_division=0))
        return {"modelo": nome_modelo, **metrics}

    def calcular_acuracia(self, modelo, nome_modelo, y_pred_train=None, y_pred_test=None):
        """Calcula a acuracia do modelo nos conjuntos de treino e teste."""
        if y_pred_train is None:
            y_pred_train = modelo.predict(self.X_train)
        if y_pred_test is None:
            y_pred_test = modelo.predict(self.X_test)

        accuracy_train = accuracy_score(self.y_train, y_pred_train)
        accuracy_test = accuracy_score(self.y_test, y_pred_test)
        print(f"\nACURACIA - {nome_modelo}:")
        print(f"  - Treinamento: {accuracy_train:.4f} ({accuracy_train:.2%})")
        print(f"  - Teste: {accuracy_test:.4f} ({accuracy_test:.2%})")
        return accuracy_train, accuracy_test

    def comparar_modelos(self, resultados_dt, resultados_lr):
        print("\n" + "-" * 45)
        print("COMPARACAO E DISCUSSAO DAS METRICAS")
        print("-" * 45)
        print(f"\n{'METRICA':<20} {'ARVORE DECISAO':<20} {'REG. LOGISTICA':<20}")
        print("-" * 65)
        for metric in ["accuracy_test", "recall_test", "f1_test"]:
            label = metric.replace("_test", "").title()
            print(f"{label:<20} {resultados_dt[metric]:<20.4f} {resultados_lr[metric]:<20.4f}")
        print("\n[DISCUSAO] Para triagem de PCOS, o recall ajuda a reduzir falsos negativos.")
        print("O F1-score equilibra falsos positivos e falsos negativos.")
        print("A accuracy deve ser analisada junto com essas metricas.")

    def pipeline_completo(self):
        print("\n" + "-" * 45)
        print("PIPELINE COMPLETO - CLASSIFICACAO DE PCOS")
        print("-" * 45)
        self.carregar_dados()
        self.explorar_dados()
        X, y = self.preprocessar_dados()
        self.dividir_dados(X, y)
        self.treinar_arvore_decisao()
        self.treinar_regressao_logistica()
        resultados_dt = self.avaliar_modelo(self.dt_model, "Arvore de Decisao")
        resultados_lr = self.avaliar_modelo(self.lr_model, "Regressao Logistica")
        self.comparar_modelos(resultados_dt, resultados_lr)
        print("\nACURACIA FINAL DOS MODELOS (TESTE):")
        print(f"  - Arvore de Decisao: {resultados_dt['accuracy_test']:.4f} ({resultados_dt['accuracy_test']:.2%})")
        print(f"  - Regressao Logistica: {resultados_lr['accuracy_test']:.4f} ({resultados_lr['accuracy_test']:.2%})")
        print("\n" + "-" * 45)
        print("AWEEE PIPELINE FINALIZADO COM SUCESSO!")
        print("-" * 45)


def main():
    project_root = Path(__file__).resolve().parents[2]
    filepath = project_root / "base_dados_tratada" / "PCOS_unificado.csv"
    PCOSDataAnalysis(filepath).pipeline_completo()


if __name__ == "__main__":
    main()