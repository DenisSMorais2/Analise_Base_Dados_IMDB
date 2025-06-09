"""
Análise Exploratória dos Dados IMDB - Versão Scikit-Learn
========================================================

Script para análise exploratória completa do dataset IMDB
sem dependências pesadas como TensorFlow.

Análises incluídas:
- Estatísticas básicas do dataset
- Distribuição de classes
- Análise de comprimento de texto
- Palavras mais frequentes
- Visualizações informativas
- Preparação dos dados
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from wordcloud import WordCloud
import warnings

warnings.filterwarnings('ignore')

# Configurações de visualização
plt.style.use('default')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 11
sns.set_palette("husl")

class IMDBDataExplorer:
    """Classe para análise exploratória do dataset IMDB"""
    
    def __init__(self):
        self.df = None
        
    def load_data(self, file_path, sample_size=None):
        """Carregar dados do arquivo CSV"""
        print("📊 Carregando dataset IMDB...")
        
        try:
            self.df = pd.read_csv(file_path)
            print(f"✅ Dataset carregado com sucesso!")
            print(f"📈 Forma do dataset: {self.df.shape}")
            
            if sample_size and sample_size < len(self.df):
                self.df = self.df.sample(n=sample_size, random_state=42)
                print(f"📋 Usando amostra de {sample_size:,} registros")
                
        except FileNotFoundError:
            print(f"❌ Arquivo não encontrado: {file_path}")
            print("💡 Certifique-se de que o arquivo está no diretório correto")
            return False
        except Exception as e:
            print(f"❌ Erro ao carregar arquivo: {e}")
            return False
            
        return True
    
    def basic_info(self):
        """Informações básicas do dataset"""
        if self.df is None:
            print("❌ Dataset não carregado")
            return
            
        print("\n" + "="*60)
        print("📋 INFORMAÇÕES BÁSICAS DO DATASET")
        print("="*60)
        
        print(f"📊 Forma do dataset: {self.df.shape}")
        print(f"📝 Colunas: {list(self.df.columns)}")
        print(f"🗂️ Tipos de dados:")
        print(self.df.dtypes)
        
        print(f"\n🔍 Informações gerais:")
        print(self.df.info())
        
        print(f"\n❓ Valores faltantes:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("✅ Nenhum valor faltante encontrado!")
        else:
            print(missing)
        
        print(f"\n👁️ Primeiras 3 linhas:")
        print(self.df.head(3))
        
    def sentiment_analysis(self):
        """Análise da distribuição de sentimentos"""
        if self.df is None:
            return
            
        print("\n" + "="*60)
        print("😊 ANÁLISE DE SENTIMENTOS")
        print("="*60)
        
        # Contagem de sentimentos
        sentiment_counts = self.df['sentiment'].value_counts()
        print(f"📊 Distribuição de sentimentos:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"   {sentiment}: {count:,} ({percentage:.1f}%)")
        
        # Verificar balanceamento
        balance_ratio = min(sentiment_counts) / max(sentiment_counts)
        print(f"\n⚖️ Ratio de balanceamento: {balance_ratio:.3f}")
        
        if balance_ratio > 0.9:
            print("✅ Dataset bem balanceado!")
        elif balance_ratio > 0.7:
            print("⚠️ Dataset moderadamente balanceado")
        else:
            print("❌ Dataset desbalanceado - considere técnicas de balanceamento")
    
    def text_statistics(self):
        """Estatísticas do texto"""
        if self.df is None:
            return
            
        print("\n" + "="*60)
        print("📝 ESTATÍSTICAS DO TEXTO")
        print("="*60)
        
        # Calcular estatísticas
        self.df['text_length'] = self.df['review'].str.len()
        self.df['word_count'] = self.df['review'].str.split().str.len()
        self.df['avg_word_length'] = self.df['review'].apply(
            lambda x: np.mean([len(word) for word in x.split()])
        )
        
        # Estatísticas descritivas
        print("📏 Comprimento do texto (caracteres):")
        print(self.df['text_length'].describe())
        
        print("\n📚 Número de palavras:")
        print(self.df['word_count'].describe())
        
        print("\n📖 Comprimento médio das palavras:")
        print(self.df['avg_word_length'].describe())
        
        # Por sentimento
        print("\n📊 Estatísticas por sentimento:")
        stats_by_sentiment = self.df.groupby('sentiment')[
            ['text_length', 'word_count', 'avg_word_length']
        ].mean()
        print(stats_by_sentiment)
        
        # Identificar outliers
        text_q99 = self.df['text_length'].quantile(0.99)
        words_q99 = self.df['word_count'].quantile(0.99)
        
        print(f"\n🎯 Outliers (99º percentil):")
        print(f"   Comprimento máximo sugerido: {text_q99:.0f} caracteres")
        print(f"   Palavras máximas sugeridas: {words_q99:.0f} palavras")
        
    def clean_text_sample(self, text):
        """Função auxiliar para limpeza de texto"""
        # Remover HTML
        text = re.sub(r'<[^>]+>', '', text)
        # Manter apenas letras e espaços
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Converter para minúsculas
        text = text.lower()
        # Remover espaços extras
        text = re.sub(r'\s+', ' ', text).strip()
        return text
        
    def word_frequency_analysis(self, top_n=20):
        """Análise de frequência de palavras"""
        if self.df is None:
            return
            
        print("\n" + "="*60)
        print(f"🔤 ANÁLISE DE FREQUÊNCIA DE PALAVRAS (Top {top_n})")
        print("="*60)
        
        # Stop words básicas em inglês
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 
            'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their'
        }
        
        # Separar por sentimento
        positive_reviews = self.df[self.df['sentiment'] == 'positive']['review']
        negative_reviews = self.df[self.df['sentiment'] == 'negative']['review']
        
        # Extrair palavras
        def extract_words(reviews):
            all_words = []
            for review in reviews:
                cleaned = self.clean_text_sample(review)
                words = [word for word in cleaned.split() 
                        if word not in stop_words and len(word) > 2]
                all_words.extend(words)
            return all_words
        
        print("🔄 Processando palavras...")
        positive_words = extract_words(positive_reviews)
        negative_words = extract_words(negative_reviews)
        
        # Contar frequências
        positive_freq = Counter(positive_words).most_common(top_n)
        negative_freq = Counter(negative_words).most_common(top_n)
        
        print(f"\n✅ Top {top_n} palavras em reviews POSITIVOS:")
        for i, (word, count) in enumerate(positive_freq, 1):
            print(f"   {i:2d}. {word:<15} ({count:,} vezes)")
        
        print(f"\n❌ Top {top_n} palavras em reviews NEGATIVOS:")
        for i, (word, count) in enumerate(negative_freq, 1):
            print(f"   {i:2d}. {word:<15} ({count:,} vezes)")
        
        return positive_freq, negative_freq
    
    def create_visualizations(self, save_path='imdb_exploration.png'):
        """Criar visualizações exploratórias"""
        if self.df is None:
            return
            
        print(f"\n📊 Criando visualizações...")
        
        # Configurar subplot
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        fig.suptitle('Análise Exploratória - Dataset IMDB', fontsize=16, fontweight='bold')
        
        # 1. Distribuição de sentimentos
        sentiment_counts = self.df['sentiment'].value_counts()
        colors = ['lightgreen', 'lightcoral']
        axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                      autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0, 0].set_title('Distribuição de Sentimentos')
        
        # 2. Histograma de comprimento de texto
        axes[0, 1].hist([
            self.df[self.df['sentiment'] == 'positive']['text_length'],
            self.df[self.df['sentiment'] == 'negative']['text_length']
        ], bins=50, alpha=0.7, label=['Positivo', 'Negativo'], color=['green', 'red'])
        axes[0, 1].set_xlabel('Comprimento do Texto (caracteres)')
        axes[0, 1].set_ylabel('Frequência')
        axes[0, 1].set_title('Distribuição do Comprimento do Texto')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Boxplot de número de palavras
        sentiment_data = [
            self.df[self.df['sentiment'] == 'positive']['word_count'],
            self.df[self.df['sentiment'] == 'negative']['word_count']
        ]
        axes[0, 2].boxplot(sentiment_data, labels=['Positivo', 'Negativo'], patch_artist=True)
        axes[0, 2].set_ylabel('Número de Palavras')
        axes[0, 2].set_title('Distribuição de Palavras por Sentimento')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Comprimento médio das palavras
        axes[1, 0].hist([
            self.df[self.df['sentiment'] == 'positive']['avg_word_length'],
            self.df[self.df['sentiment'] == 'negative']['avg_word_length']
        ], bins=30, alpha=0.7, label=['Positivo', 'Negativo'], color=['green', 'red'])
        axes[1, 0].set_xlabel('Comprimento Médio das Palavras')
        axes[1, 0].set_ylabel('Frequência')
        axes[1, 0].set_title('Distribuição do Comprimento Médio das Palavras')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Scatter plot: palavras vs caracteres
        sample_df = self.df.sample(n=min(1000, len(self.df)), random_state=42)
        colors_map = {'positive': 'green', 'negative': 'red'}
        for sentiment in ['positive', 'negative']:
            data = sample_df[sample_df['sentiment'] == sentiment]
            axes[1, 1].scatter(data['word_count'], data['text_length'], 
                             alpha=0.6, label=sentiment, c=colors_map[sentiment])
        
        axes[1, 1].set_xlabel('Número de Palavras')
        axes[1, 1].set_ylabel('Comprimento do Texto')
        axes[1, 1].set_title('Relação: Palavras vs Caracteres')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Estatísticas comparativas
        stats_comparison = self.df.groupby('sentiment')[
            ['text_length', 'word_count', 'avg_word_length']
        ].mean()
        
        x = np.arange(len(stats_comparison.columns))
        width = 0.35
        
        axes[1, 2].bar(x - width/2, stats_comparison.loc['positive'], 
                      width, label='Positivo', color='green', alpha=0.7)
        axes[1, 2].bar(x + width/2, stats_comparison.loc['negative'], 
                      width, label='Negativo', color='red', alpha=0.7)
        
        axes[1, 2].set_xlabel('Métricas')
        axes[1, 2].set_ylabel('Valores Médios')
        axes[1, 2].set_title('Comparação de Métricas por Sentimento')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(['Comprimento', 'N° Palavras', 'Tam. Médio'])
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # 7-8. Palavras mais frequentes (se disponível)
        try:
            positive_freq, negative_freq = self.word_frequency_analysis(top_n=15)
            
            # Top palavras positivas
            pos_words, pos_counts = zip(*positive_freq[:10])
            axes[2, 0].barh(range(len(pos_words)), pos_counts, color='green', alpha=0.7)
            axes[2, 0].set_yticks(range(len(pos_words)))
            axes[2, 0].set_yticklabels(pos_words)
            axes[2, 0].set_xlabel('Frequência')
            axes[2, 0].set_title('Top 10 Palavras - Reviews Positivos')
            axes[2, 0].invert_yaxis()
            axes[2, 0].grid(True, alpha=0.3)
            
            # Top palavras negativas
            neg_words, neg_counts = zip(*negative_freq[:10])
            axes[2, 1].barh(range(len(neg_words)), neg_counts, color='red', alpha=0.7)
            axes[2, 1].set_yticks(range(len(neg_words)))
            axes[2, 1].set_yticklabels(neg_words)
            axes[2, 1].set_xlabel('Frequência')
            axes[2, 1].set_title('Top 10 Palavras - Reviews Negativos')
            axes[2, 1].invert_yaxis()
            axes[2, 1].grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"⚠️ Erro na análise de palavras: {e}")
            axes[2, 0].text(0.5, 0.5, 'Análise de palavras\nnão disponível', 
                           ha='center', va='center', transform=axes[2, 0].transAxes)
            axes[2, 1].text(0.5, 0.5, 'Análise de palavras\nnão disponível', 
                           ha='center', va='center', transform=axes[2, 1].transAxes)
        
        # 9. Distribuição de comprimentos (zoom)
        # Focar nos 95% centrais para melhor visualização
        q05 = self.df['text_length'].quantile(0.05)
        q95 = self.df['text_length'].quantile(0.95)
        filtered_df = self.df[(self.df['text_length'] >= q05) & (self.df['text_length'] <= q95)]
        
        axes[2, 2].hist([
            filtered_df[filtered_df['sentiment'] == 'positive']['text_length'],
            filtered_df[filtered_df['sentiment'] == 'negative']['text_length']
        ], bins=40, alpha=0.7, label=['Positivo', 'Negativo'], color=['green', 'red'])
        axes[2, 2].set_xlabel('Comprimento do Texto (caracteres)')
        axes[2, 2].set_ylabel('Frequência')
        axes[2, 2].set_title(f'Distribuição de Comprimento (P5-P95)')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Visualizações salvas: {save_path}")
    
    def sample_reviews(self, n_samples=5):
        """Mostrar amostras de reviews"""
        if self.df is None:
            return
            
        print("\n" + "="*60)
        print(f"📖 AMOSTRAS DE REVIEWS ({n_samples} de cada sentimento)")
        print("="*60)
        
        for sentiment in ['positive', 'negative']:
            print(f"\n{'✅' if sentiment == 'positive' else '❌'} Reviews {sentiment.upper()}:")
            samples = self.df[self.df['sentiment'] == sentiment].sample(n=n_samples, random_state=42)
            
            for i, (_, row) in enumerate(samples.iterrows(), 1):
                review = row['review']
                # Mostrar apenas os primeiros 200 caracteres
                preview = review[:200] + "..." if len(review) > 200 else review
                print(f"\n   {i}. {preview}")
                print(f"      Comprimento: {len(review)} caracteres, {len(review.split())} palavras")
    
    def generate_summary_report(self):
        """Gerar relatório resumo"""
        if self.df is None:
            return
            
        print("\n" + "="*60)
        print("📋 RELATÓRIO RESUMO")
        print("="*60)
        
        # Estatísticas gerais
        total_samples = len(self.df)
        sentiment_dist = self.df['sentiment'].value_counts()
        
        print(f"📊 Dataset Overview:")
        print(f"   • Total de amostras: {total_samples:,}")
        print(f"   • Reviews positivos: {sentiment_dist.get('positive', 0):,}")
        print(f"   • Reviews negativos: {sentiment_dist.get('negative', 0):,}")
        
        # Estatísticas de texto
        mean_length = self.df['text_length'].mean()
        mean_words = self.df['word_count'].mean()
        
        print(f"\n📝 Estatísticas de Texto:")
        print(f"   • Comprimento médio: {mean_length:.0f} caracteres")
        print(f"   • Palavras médias: {mean_words:.0f} palavras")
        print(f"   • Comprimento min/max: {self.df['text_length'].min()}/{self.df['text_length'].max()}")
        
        # Recomendações para ML
        print(f"\n💡 Recomendações para Machine Learning:")
        
        # Vocabulário estimado
        sample_text = ' '.join(self.df['review'].sample(n=min(1000, len(self.df)), random_state=42))
        unique_words = len(set(re.findall(r'\b\w+\b', sample_text.lower())))
        vocab_estimate = unique_words * (len(self.df) / min(1000, len(self.df)))
        
        print(f"   • Vocabulário estimado: {vocab_estimate:,.0f} palavras únicas")
        print(f"   • Max features sugerido: {min(10000, vocab_estimate):,.0f}")
        
        # Sugestões de pré-processamento
        q95_length = self.df['text_length'].quantile(0.95)
        q95_words = self.df['word_count'].quantile(0.95)
        
        print(f"   • Comprimento máximo sugerido: {q95_length:.0f} caracteres")
        print(f"   • Número máximo de palavras: {q95_words:.0f}")
        
        # Balanceamento
        balance_ratio = min(sentiment_dist) / max(sentiment_dist)
        if balance_ratio > 0.9:
            print(f"   • ✅ Dataset balanceado (ratio: {balance_ratio:.3f})")
        else:
            print(f"   • ⚠️ Considerar balanceamento (ratio: {balance_ratio:.3f})")

def main():
    """Função principal para análise exploratória"""
    print("🔍 === ANÁLISE EXPLORATÓRIA - DATASET IMDB ===")
    print("Versão Scikit-Learn - Rápida e eficiente!\n")
    
    # Configurações
    DATA_PATH = 'IMDBDataset.csv'  # Ajuste conforme necessário
    SAMPLE_SIZE = 10000  # Use None para dataset completo
    
    # Inicializar explorador
    explorer = IMDBDataExplorer()
    
    # Carregar dados
    if not explorer.load_data(DATA_PATH, sample_size=SAMPLE_SIZE):
        print("❌ Falha ao carregar dados. Verifique o caminho do arquivo.")
        return
    
    # Executar análises
    try:
        explorer.basic_info()
        explorer.sentiment_analysis()
        explorer.text_statistics()
        explorer.sample_reviews(n_samples=3)
        explorer.create_visualizations('imdb_exploration_sklearn.png')
        explorer.generate_summary_report()
        
        print("\n" + "="*60)
        print("✅ Análise exploratória concluída com sucesso!")
        print("📁 Arquivo gerado: imdb_exploration_sklearn.png")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Erro durante a análise: {e}")
        print("💡 Verifique se todas as dependências estão instaladas:")
        print("   pip install pandas numpy matplotlib seaborn")

if __name__ == "__main__":
    main()