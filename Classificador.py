"""
Classificador de Sentimentos IMDB - Versão Scikit-Learn
=====================================================
Algoritmos utilizados:
- Logistic Regression
- Random Forest  
- Support Vector Machine (SVM)
- Gradient Boosting
- Ensemble Voting

Técnicas de NLP:
- TF-IDF Vectorization
- N-grams (1-3)
- Feature Selection
- Text Preprocessing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import time
import warnings
from collections import Counter

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve
)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2

warnings.filterwarnings('ignore')

# Configurações
plt.style.use('default')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 11
np.random.seed(42)

class IMDBClassifierSklearn:
    """Classificador de sentimentos IMDB usando Scikit-Learn"""
    
    def __init__(self):
        self.vectorizer = None
        self.models = {}
        self.results = {}
        self.best_model = None
        
    def clean_text(self, text):
        """Limpeza e pré-processamento de texto"""
        # Remover tags HTML
        text = re.sub(r'<[^>]+>', '', text)
        
        # Expandir contrações comuns
        contractions = {
            "don't": "do not", "won't": "will not", "can't": "cannot",
            "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remover caracteres especiais mas manter espaços
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Converter para minúsculas
        text = text.lower()
        
        # Remover espaços extras
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def load_and_preprocess_data(self, data_path, sample_size=None):
        """Carregar e preprocessar dados"""
        print("📊 Carregando dataset IMDB...")
        
        try:
            df = pd.read_csv(data_path)
        except FileNotFoundError:
            print("❌ Arquivo não encontrado. Certifique-se que o arquivo está no caminho correto.")
            return None
        
        print(f"✅ Dataset carregado: {df.shape}")
        
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            print(f"📋 Usando amostra de {sample_size:,} registros")
        
        # Verificar estrutura
        print(f"📈 Colunas: {list(df.columns)}")
        print(f"📊 Distribuição:")
        print(df['sentiment'].value_counts())
        
        # Limpeza de texto
        print("🧹 Aplicando limpeza de texto...")
        df['cleaned_review'] = df['review'].apply(self.clean_text)
        
        # Converter labels
        df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
        
        # Estatísticas
        df['text_length'] = df['cleaned_review'].str.len()
        df['word_count'] = df['cleaned_review'].str.split().str.len()
        
        print(f"📝 Estatísticas do texto:")
        print(f"   Comprimento médio: {df['text_length'].mean():.1f} caracteres")
        print(f"   Palavras médias: {df['word_count'].mean():.1f} palavras")
        print(f"   Comprimento min/max: {df['text_length'].min()}/{df['text_length'].max()}")
        
        return df
    
    def create_features(self, texts, max_features=10000, ngram_range=(1, 2)):
        """Criar features usando TF-IDF"""
        print(f"🔧 Criando features TF-IDF...")
        print(f"   Max features: {max_features:,}")
        print(f"   N-grams: {ngram_range}")
        
        # Configurar TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            max_df=0.95,  # Ignorar palavras muito comuns
            min_df=2,     # Ignorar palavras muito raras
            sublinear_tf=True  # Aplicar escala logarítmica
        )
        
        # Ajustar e transformar
        features = self.vectorizer.fit_transform(texts)
        
        print(f"✅ Features criadas: {features.shape}")
        print(f"📚 Vocabulário: {len(self.vectorizer.vocabulary_):,} palavras")
        
        return features
    
    def build_models(self):
        """Construir múltiplos modelos para comparação"""
        print("🏗️ Construindo modelos...")
        
        models = {
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                solver='liblinear'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            'svm': SVC(
                kernel='linear',
                random_state=42,
                probability=True
            ),
            'naive_bayes': MultinomialNB(
                alpha=1.0
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            )
        }
        
        print(f"📦 Modelos criados: {list(models.keys())}")
        return models
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Treinar e avaliar todos os modelos"""
        print("\n🚀 Iniciando treinamento e avaliação...")
        
        models = self.build_models()
        
        for name, model in models.items():
            print(f"\n📈 Treinando {name}...")
            
            # Treinar
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Predições
            start_time = time.time()
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            prediction_time = time.time() - start_time
            
            # Métricas
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # Armazenar resultados
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'training_time': training_time,
                'prediction_time': prediction_time,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"   ✅ Accuracy: {accuracy:.4f}")
            print(f"   📊 F1-Score: {f1:.4f}")
            print(f"   ⏱️ Tempo treino: {training_time:.2f}s")
            print(f"   ⚡ Tempo predição: {prediction_time:.4f}s")
        
        # Identificar melhor modelo
        best_name = max(self.results.keys(), key=lambda k: self.results[k]['f1_score'])
        self.best_model = self.results[best_name]['model']
        
        print(f"\n🏆 Melhor modelo: {best_name} (F1: {self.results[best_name]['f1_score']:.4f})")
        return self.results
    
    def create_ensemble(self, X_train, y_train):
        """Criar ensemble dos melhores modelos"""
        print("\n🤝 Criando ensemble...")
        
        # Selecionar top 3 modelos por F1-score
        top_models = sorted(self.results.items(), 
                          key=lambda x: x[1]['f1_score'], 
                          reverse=True)[:3]
        
        estimators = [(name, results['model']) for name, results in top_models]
        
        # Criar voting classifier
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft'  # Usar probabilidades
        )
        
        # Treinar ensemble
        start_time = time.time()
        ensemble.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"✅ Ensemble criado com: {[name for name, _ in estimators]}")
        print(f"⏱️ Tempo de treinamento: {training_time:.2f}s")
        
        return ensemble
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='logistic_regression'):
        """Otimização de hiperparâmetros"""
        print(f"\n🔧 Otimizando hiperparâmetros para {model_name}...")
        
        if model_name == 'logistic_regression':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
            model = LogisticRegression(random_state=42, max_iter=1000)
            
        elif model_name == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
            
        elif model_name == 'svm':
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
            model = SVC(random_state=42, probability=True)
        
        else:
            print(f"❌ Modelo {model_name} não suportado para tuning")
            return None
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, 
            cv=3, scoring='f1', 
            n_jobs=-1, verbose=1
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        tuning_time = time.time() - start_time
        
        print(f"✅ Melhor score: {grid_search.best_score_:.4f}")
        print(f"⚙️ Melhores parâmetros: {grid_search.best_params_}")
        print(f"⏱️ Tempo de tuning: {tuning_time:.2f}s")
        
        return grid_search.best_estimator_
    
    def analyze_features(self, top_n=20):
        """Analisar features mais importantes"""
        print(f"\n🔍 Analisando top {top_n} features...")
        
        if not self.best_model or not self.vectorizer:
            print("❌ Modelo ou vectorizer não disponível")
            return None
        
        # Obter nomes das features
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Obter importâncias (depende do modelo)
        if hasattr(self.best_model, 'coef_'):
            # Modelos lineares (Logistic Regression, SVM)
            importances = self.best_model.coef_[0]
            
        elif hasattr(self.best_model, 'feature_importances_'):
            # Modelos baseados em árvore (Random Forest, Gradient Boosting)
            importances = self.best_model.feature_importances_
            
        else:
            print("❌ Modelo não suporta análise de importância")
            return None
        
        # Criar DataFrame com importâncias
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(importances),
            'coefficient': importances
        }).sort_values('importance', ascending=False)
        
        print("🔥 Top features positivas (sentimento positivo):")
        positive_features = feature_importance[feature_importance['coefficient'] > 0].head(top_n//2)
        for idx, row in positive_features.iterrows():
            print(f"   📈 {row['feature']}: {row['coefficient']:.4f}")
        
        print("\n❄️ Top features negativas (sentimento negativo):")
        negative_features = feature_importance[feature_importance['coefficient'] < 0].head(top_n//2)
        for idx, row in negative_features.iterrows():
            print(f"   📉 {row['feature']}: {row['coefficient']:.4f}")
        
        return feature_importance
    
    def predict_sentiment(self, text, return_proba=False):
        """Predizer sentimento de um texto"""
        if not self.best_model or not self.vectorizer:
            print("❌ Modelo não treinado")
            return None
        
        # Limpar e vetorizar texto
        cleaned = self.clean_text(text)
        features = self.vectorizer.transform([cleaned])
        
        # Predição
        prediction = self.best_model.predict(features)[0]
        
        if hasattr(self.best_model, 'predict_proba'):
            probabilities = self.best_model.predict_proba(features)[0]
            confidence = max(probabilities)
            prob_positive = probabilities[1]
        else:
            confidence = 1.0  # Para modelos sem probabilidade
            prob_positive = float(prediction)
        
        sentiment = "Positivo" if prediction == 1 else "Negativo"
        
        if return_proba:
            return sentiment, confidence, prob_positive
        else:
            return sentiment, confidence
    
    def create_visualizations(self, y_test, save_path='sklearn_results.png'):
        """Criar visualizações dos resultados"""
        print("\n📊 Criando visualizações...")
        
        n_models = len(self.results)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Resultados do Classificador IMDB - Scikit-Learn', fontsize=16, fontweight='bold')
        
        # 1. Comparação de métricas
        model_names = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        
        x = np.arange(len(model_names))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            values = [self.results[name][metric] for name in model_names]
            axes[0, 0].bar(x + i*width, values, width, label=metric, alpha=0.8)
        
        axes[0, 0].set_xlabel('Modelos')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Comparação de Métricas')
        axes[0, 0].set_xticks(x + width * 2)
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Tempo de treinamento
        training_times = [self.results[name]['training_time'] for name in model_names]
        bars = axes[0, 1].bar(model_names, training_times, color='skyblue', alpha=0.7)
        axes[0, 1].set_ylabel('Tempo (segundos)')
        axes[0, 1].set_title('Tempo de Treinamento')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for bar, time_val in zip(bars, training_times):
            axes[0, 1].annotate(f'{time_val:.2f}s',
                              xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                              xytext=(0, 3), textcoords='offset points',
                              ha='center', va='bottom', fontsize=9)
        
        # 3. F1-Score vs Tempo (Eficiência)
        f1_scores = [self.results[name]['f1_score'] for name in model_names]
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        
        scatter = axes[0, 2].scatter(training_times, f1_scores, c=colors, s=100, alpha=0.7)
        axes[0, 2].set_xlabel('Tempo de Treinamento (s)')
        axes[0, 2].set_ylabel('F1-Score')
        axes[0, 2].set_title('Eficiência: F1-Score vs Tempo')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Adicionar labels
        for i, name in enumerate(model_names):
            axes[0, 2].annotate(name, (training_times[i], f1_scores[i]),
                              xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # 4. Matriz de confusão do melhor modelo
        best_name = max(self.results.keys(), key=lambda k: self.results[k]['f1_score'])
        best_pred = self.results[best_name]['y_pred']
        
        cm = confusion_matrix(y_test, best_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title(f'Matriz de Confusão - {best_name}')
        axes[1, 0].set_xlabel('Predição')
        axes[1, 0].set_ylabel('Real')
        axes[1, 0].set_xticklabels(['Negativo', 'Positivo'])
        axes[1, 0].set_yticklabels(['Negativo', 'Positivo'])
        
        # 5. Curvas ROC
        for name in model_names:
            if 'y_pred_proba' in self.results[name]:
                fpr, tpr, _ = roc_curve(y_test, self.results[name]['y_pred_proba'])
                auc_score = self.results[name]['auc']
                axes[1, 1].plot(fpr, tpr, label=f'{name} (AUC={auc_score:.3f})', linewidth=2)
        
        axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1, 1].set_xlabel('Taxa de Falsos Positivos')
        axes[1, 1].set_ylabel('Taxa de Verdadeiros Positivos')
        axes[1, 1].set_title('Curvas ROC')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Distribuição de scores
        best_proba = self.results[best_name]['y_pred_proba']
        
        # Separar por classe real
        positive_scores = best_proba[y_test == 1]
        negative_scores = best_proba[y_test == 0]
        
        axes[1, 2].hist(negative_scores, bins=30, alpha=0.7, label='Negativos Reais', color='red')
        axes[1, 2].hist(positive_scores, bins=30, alpha=0.7, label='Positivos Reais', color='green')
        axes[1, 2].axvline(x=0.5, color='black', linestyle='--', label='Threshold')
        axes[1, 2].set_xlabel('Score de Probabilidade')
        axes[1, 2].set_ylabel('Frequência')
        axes[1, 2].set_title(f'Distribuição de Scores - {best_name}')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Visualizações salvas: {save_path}")
    
    def save_classifier(self, filepath):
        """Salvar classificador completo"""
        data_to_save = {
            'vectorizer': self.vectorizer,
            'best_model': self.best_model,
            'results': self.results,
            'models': {name: result['model'] for name, result in self.results.items()}
        }
        
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(data_to_save, f)
        
        print(f"💾 Classificador salvo: {filepath}.pkl")
    
    def load_classifier(self, filepath):
        """Carregar classificador"""
        try:
            with open(f"{filepath}.pkl", 'rb') as f:
                data = pickle.load(f)
            
            self.vectorizer = data['vectorizer']
            self.best_model = data['best_model']
            self.results = data['results']
            
            print(f"✅ Classificador carregado: {filepath}.pkl")
            return True
        except FileNotFoundError:
            print(f"❌ Arquivo não encontrado: {filepath}.pkl")
            return False

def main():
    """Função principal"""
    print("🎬 === CLASSIFICADOR DE SENTIMENTOS IMDB - SCIKIT-LEARN ===")
    print("Versão sem TensorFlow - Rápida e fácil de usar!\n")
    
    # Configurações
    DATA_PATH = 'IMDBDataset.csv'  # Ajuste o caminho conforme necessário
    SAMPLE_SIZE = 10000  # Use None para dataset completo, ou um número menor para teste
    
    # Inicializar classificador
    classifier = IMDBClassifierSklearn()
    
    # 1. Carregar e preprocessar dados
    print("=" * 60)
    df = classifier.load_and_preprocess_data(DATA_PATH, sample_size=SAMPLE_SIZE)
    
    if df is None:
        print("❌ Erro ao carregar dados. Verifique o caminho do arquivo.")
        return
    
    # 2. Dividir dados
    print("\n" + "=" * 60)
    print("📊 Dividindo dados...")
    
    X = df['cleaned_review'].values
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Dados divididos:")
    print(f"   Treino: {len(X_train):,} amostras")
    print(f"   Teste: {len(X_test):,} amostras")
    print(f"   Distribuição treino: {np.bincount(y_train)}")
    print(f"   Distribuição teste: {np.bincount(y_test)}")
    
    # 3. Criar features
    print("\n" + "=" * 60)
    X_train_features = classifier.create_features(X_train, max_features=10000, ngram_range=(1, 2))
    X_test_features = classifier.vectorizer.transform(X_test)
    
    # 4. Treinar e avaliar modelos
    print("\n" + "=" * 60)
    results = classifier.train_and_evaluate(X_train_features, X_test_features, y_train, y_test)
    
    # 5. Criar ensemble
    print("\n" + "=" * 60)
    ensemble = classifier.create_ensemble(X_train_features, y_train)
    
    # Avaliar ensemble
    y_pred_ensemble = ensemble.predict(X_test_features)
    y_pred_proba_ensemble = ensemble.predict_proba(X_test_features)[:, 1]
    
    ensemble_f1 = f1_score(y_test, y_pred_ensemble)
    ensemble_auc = roc_auc_score(y_test, y_pred_proba_ensemble)
    
    print(f"🤝 Performance do Ensemble:")
    print(f"   F1-Score: {ensemble_f1:.4f}")
    print(f"   AUC: {ensemble_auc:.4f}")
    
    # 6. Análise de features
    print("\n" + "=" * 60)
    feature_importance = classifier.analyze_features(top_n=20)
    
    # 7. Otimização de hiperparâmetros (opcional)
    print("\n" + "=" * 60)
    print("🔧 Deseja otimizar hiperparâmetros? (pode demorar)")
    # tuned_model = classifier.hyperparameter_tuning(X_train_features, y_train, 'logistic_regression')
    
    # 8. Teste com exemplos
    print("\n" + "=" * 60)
    print("🧪 Testando com exemplos...")
    
    test_examples = [
        "This movie is absolutely fantastic! Amazing acting and great story.",
        "Terrible film, complete waste of time. Very boring and poorly made.",
        "The movie was okay, nothing special but watchable.",
        "One of the best films I've ever seen! Incredible cinematography.",
        "I fell asleep halfway through. Very disappointing and overrated.",
        "Mixed feelings about this one. Good visuals but weak plot.",
        "A masterpiece of cinema! Every scene is perfectly crafted.",
        "Predictable storyline with mediocre performances."
    ]
    
    print("🎯 Resultados das predições:")
    for i, example in enumerate(test_examples, 1):
        sentiment, confidence, prob = classifier.predict_sentiment(example, return_proba=True)
        print(f"{i:2d}. \"{example[:60]}{'...' if len(example) > 60 else ''}\"")
        print(f"    🎯 {sentiment} (Confiança: {confidence:.3f}, Prob: {prob:.3f})")
        print()
    
    # 9. Visualizações
    print("=" * 60)
    classifier.create_visualizations(y_test, 'sklearn_imdb_results.png')
    
    # 10. Salvar modelo
    print("=" * 60)
    classifier.save_classifier('imdb_sklearn_classifier')
    
    # 11. Relatório final
    print("=" * 60)
    print("📋 === RELATÓRIO FINAL ===")
    
    best_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
    best_results = results[best_name]
    
    print(f"🏆 Melhor modelo individual: {best_name}")
    print(f"📊 Métricas do melhor modelo:")
    print(f"   • Accuracy: {best_results['accuracy']:.4f}")
    print(f"   • Precision: {best_results['precision']:.4f}")
    print(f"   • Recall: {best_results['recall']:.4f}")
    print(f"   • F1-Score: {best_results['f1_score']:.4f}")
    print(f"   • AUC: {best_results['auc']:.4f}")
    
    print(f"\n🤝 Performance do Ensemble:")
    print(f"   • F1-Score: {ensemble_f1:.4f}")
    print(f"   • AUC: {ensemble_auc:.4f}")
    
    print(f"\n⏱️ Tempos de processamento:")
    total_training_time = sum(result['training_time'] for result in results.values())
    print(f"   • Treinamento total: {total_training_time:.2f}s")
    print(f"   • Predição média: {np.mean([result['prediction_time'] for result in results.values()])*1000:.1f}ms")
    
    print(f"\n📁 Arquivos gerados:")
    print(f"   • sklearn_imdb_results.png - Visualizações detalhadas")
    print(f"   • imdb_sklearn_classifier.pkl - Modelo treinado")
    
    print(f"\n✅ Processamento concluído com sucesso!")
    print("=" * 60)
    
    return classifier

if __name__ == "__main__":
    classifier = main()