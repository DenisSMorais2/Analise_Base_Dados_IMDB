"""
Teste Rápido - Classificador IMDB Sklearn
========================================

Script para teste rápido do classificador sem TensorFlow.
Ideal para verificar se tudo está funcionando corretamente.

Funcionalidades:
- Teste com dados sintéticos (se não tiver o arquivo CSV)
- Treinamento rápido com amostra pequena
- Validação básica do pipeline
- Exemplo de uso prático
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings

warnings.filterwarnings('ignore')

def create_synthetic_data(n_samples=1000):
    """Criar dados sintéticos para teste"""
    print("🔧 Criando dados sintéticos para teste...")
    
    # Palavras positivas e negativas
    positive_words = [
        "amazing", "excellent", "fantastic", "wonderful", "brilliant", "outstanding",
        "perfect", "incredible", "awesome", "superb", "magnificent", "terrific",
        "great", "good", "love", "beautiful", "impressive", "remarkable"
    ]
    
    negative_words = [
        "terrible", "awful", "horrible", "disgusting", "pathetic", "worst",
        "boring", "stupid", "annoying", "disappointing", "waste", "bad",
        "hate", "ugly", "ridiculous", "useless", "painful", "dreadful"
    ]
    
    neutral_words = [
        "movie", "film", "story", "plot", "character", "actor", "scene",
        "director", "script", "cinema", "watch", "see", "time", "end",
        "beginning", "middle", "part", "moment", "way", "thing"
    ]
    
    reviews = []
    sentiments = []
    
    for i in range(n_samples):
        # Determinar sentimento
        sentiment = np.random.choice(['positive', 'negative'])
        
        # Criar review
        if sentiment == 'positive':
            # Mais palavras positivas
            words = (np.random.choice(positive_words, size=np.random.randint(2, 4)).tolist() +
                    np.random.choice(neutral_words, size=np.random.randint(3, 8)).tolist())
        else:
            # Mais palavras negativas
            words = (np.random.choice(negative_words, size=np.random.randint(2, 4)).tolist() +
                    np.random.choice(neutral_words, size=np.random.randint(3, 8)).tolist())
        
        # Embaralhar palavras e criar frase
        np.random.shuffle(words)
        review = "This " + " ".join(words) + " and I think it was really something."
        
        reviews.append(review)
        sentiments.append(sentiment)
    
    df = pd.DataFrame({
        'review': reviews,
        'sentiment': sentiments
    })
    
    print(f"✅ Dados sintéticos criados: {len(df)} amostras")
    print(f"📊 Distribuição: {df['sentiment'].value_counts().to_dict()}")
    
    return df

def clean_text(text):
    """Função simples de limpeza de texto"""
    # Remover caracteres especiais
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Converter para minúsculas
    text = text.lower()
    # Remover espaços extras
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def quick_classification_test(df):
    """Teste rápido de classificação"""
    print("\n🚀 Iniciando teste rápido de classificação...")
    
    # Preparar dados
    print("📊 Preparando dados...")
    X = df['review'].apply(clean_text)
    y = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Dados divididos: {len(X_train)} treino, {len(X_test)} teste")
    
    # Vetorização TF-IDF
    print("🔧 Criando features TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"📈 Features criadas: {X_train_tfidf.shape}")
    
    # Testar múltiplos modelos
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n📈 Testando {name}...")
        
        # Treinar
        start_time = time.time()
        model.fit(X_train_tfidf, y_train)
        train_time = time.time() - start_time
        
        # Predizer
        start_time = time.time()
        y_pred = model.predict(X_test_tfidf)
        pred_time = time.time() - start_time
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'train_time': train_time,
            'pred_time': pred_time,
            'model': model
        }
        
        print(f"   ✅ Accuracy: {accuracy:.3f}")
        print(f"   📊 F1-Score: {f1:.3f}")
        print(f"   ⏱️ Tempo treino: {train_time:.3f}s")
        print(f"   ⚡ Tempo predição: {pred_time:.4f}s")
    
    # Melhor modelo
    best_model_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
    best_model = results[best_model_name]['model']
    
    print(f"\n🏆 Melhor modelo: {best_model_name}")
    
    return vectorizer, best_model, results

def test_examples(vectorizer, model):
    """Testar com exemplos práticos"""
    print("\n🧪 Testando com exemplos práticos...")
    
    test_examples = [
        "This movie is absolutely amazing and fantastic!",
        "Terrible film, completely boring and awful.",
        "The movie was okay, nothing too special.",
        "Incredible story with outstanding performances.",
        "Waste of time, very disappointing experience.",
        "Good entertainment but not the best.",
        "Brilliant cinematography and excellent acting!",
        "Boring plot with terrible characters."
    ]
    
    print("🎯 Resultados das predições:")
    
    for i, text in enumerate(test_examples, 1):
        # Preprocessar
        cleaned = clean_text(text)
        features = vectorizer.transform([cleaned])
        
        # Predizer
        prediction = model.predict(features)[0]
        
        # Probabilidade (se disponível)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            confidence = max(proba)
        else:
            confidence = 1.0
        
        sentiment = "Positivo" if prediction == 1 else "Negativo"
        
        print(f"{i:2d}. \"{text}\"")
        print(f"    🎯 {sentiment} (Confiança: {confidence:.3f})")
        print()

def create_simple_visualization(results, save_path='quick_test_results.png'):
    """Criar visualização simples dos resultados"""
    print("📊 Criando visualização...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Comparação de métricas
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    f1_scores = [results[model]['f1_score'] for model in models]
    
    x = range(len(models))
    width = 0.35
    
    ax1.bar([i - width/2 for i in x], accuracies, width, label='Accuracy', alpha=0.8)
    ax1.bar([i + width/2 for i in x], f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax1.set_xlabel('Modelos')
    ax1.set_ylabel('Score')
    ax1.set_title('Comparação de Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Tempo de treinamento
    train_times = [results[model]['train_time'] for model in models]
    bars = ax2.bar(models, train_times, color='lightblue', alpha=0.7)
    
    ax2.set_ylabel('Tempo (segundos)')
    ax2.set_title('Tempo de Treinamento')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Adicionar valores nas barras
    for bar, time_val in zip(bars, train_times):
        ax2.annotate(f'{time_val:.3f}s',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Visualização salva: {save_path}")

def main():
    """Função principal do teste rápido"""
    print("⚡ === TESTE RÁPIDO - CLASSIFICADOR IMDB SKLEARN ===")
    print("Versão leve e rápida para verificação!\n")
    
    # Tentar carregar dados reais primeiro
    try:
        print("📊 Tentando carregar dados reais...")
        df = pd.read_csv('IMDBDataset.csv')
        
        # Usar amostra pequena para teste rápido
        sample_size = 2000
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            print(f"📋 Usando amostra de {sample_size} registros para teste rápido")
        
        print(f"✅ Dados reais carregados: {df.shape}")
        
    except FileNotFoundError:
        print("⚠️ Arquivo IMDBDataset.csv não encontrado")
        print("🔧 Usando dados sintéticos para demonstração...")
        df = create_synthetic_data(n_samples=2000)
    
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        print("🔧 Usando dados sintéticos para demonstração...")
        df = create_synthetic_data(n_samples=2000)
    
    # Executar teste de classificação
    try:
        vectorizer, best_model, results = quick_classification_test(df)
        
        # Testar com exemplos
        test_examples(vectorizer, best_model)
        
        # Criar visualização
        create_simple_visualization(results)
        
        # Relatório final
        print("=" * 60)
        print("📋 RELATÓRIO DO TESTE RÁPIDO")
        print("=" * 60)
        
        print("🏆 Resultados por modelo:")
        for name, result in results.items():
            print(f"\n📈 {name}:")
            print(f"   • Accuracy: {result['accuracy']:.3f}")
            print(f"   • F1-Score: {result['f1_score']:.3f}")
            print(f"   • Tempo treino: {result['train_time']:.3f}s")
            print(f"   • Tempo predição: {result['pred_time']:.4f}s")
        
        best_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
        print(f"\n🎯 Melhor modelo: {best_name}")
        print(f"📊 F1-Score: {results[best_name]['f1_score']:.3f}")
        
        print(f"\n📁 Arquivos gerados:")
        print(f"   • quick_test_results.png - Visualização dos resultados")
        
        print(f"\n✅ Teste rápido concluído com sucesso!")
        print("💡 Para usar o classificador completo, execute: python imdb_classifier_sklearn.py")
        
    except Exception as e:
        print(f"❌ Erro durante o teste: {e}")
        print("\n🔧 Verificações necessárias:")
        print("   1. Instale as dependências: pip install pandas numpy scikit-learn matplotlib")
        print("   2. Verifique se o Python está atualizado")
        print("   3. Execute em um ambiente virtual se possível")

def check_dependencies():
    """Verificar dependências necessárias"""
    print("🔍 Verificando dependências...")
    
    required_packages = {
        'pandas': 'pip install pandas',
        'numpy': 'pip install numpy', 
        'sklearn': 'pip install scikit-learn',
        'matplotlib': 'pip install matplotlib'
    }
    
    missing_packages = []
    
    for package, install_cmd in required_packages.items():
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - Execute: {install_cmd}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Instale os pacotes em falta e execute novamente")
        return False
    else:
        print(f"✅ Todas as dependências estão instaladas!")
        return True

if __name__ == "__main__":
    # Verificar dependências primeiro
    if check_dependencies():
        print()
        main()
    else:
        print("\n💡 Comando rápido para instalar tudo:")
        print("pip install pandas numpy scikit-learn matplotlib seaborn")