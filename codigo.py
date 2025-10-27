import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
from lime.lime_text import LimeTextExplainer
import json
from collections import defaultdict
from tqdm import tqdm
import gc
import re

STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has',
    'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
    'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
    'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 'its', 'our',
    'their', 'me', 'him', 'them', 'us'
}
# ============================================================
# CONFIGURAÇÕES PRINCIPAIS - AJUSTE AQUI
# ============================================================
CONFIG = {
    # Quantas amostras processar por vez
    "num_samples": 500,  # Comece com 100, depois aumente para 500, 1000, 5000, etc.
    
    # Índice inicial (para processar em lotes)
    "start_index": 1501,  # Mude para 100, 200, 300... em execuções sucessivas
    
    # Quantas features o LIME deve analisar por texto
    "lime_num_features": 10,  # Mais features = mais palavras analisadas por texto
    
    # Quantas perturbações o LIME faz por texto
    "lime_num_samples": 250,  # Menos = mais rápido, mais = mais preciso
    
    # Nome do arquivo de saída
    "output_file": "lime_results_batch.json4",  # Mude o nome entre batches
    
    # Controle de progresso
    "show_progress_every": 10,  # Mostra progresso a cada X amostras
    
    # Comprimento máximo em tokens (512 é o máximo do DistilBERT)
    "max_tokens": 512,
}

print("="*70)
print(" ANÁLISE LIME EM LARGA ESCALA - IDENTIFICAÇÃO DE VOCABULÁRIO DECISIVO")
print(" (VERSÃO CORRIGIDA)")
print("="*70)
print(f"\n📊 CONFIGURAÇÃO ATUAL:")
print(f"   • Amostras a processar: {CONFIG['num_samples']}")
print(f"   • Índice inicial: {CONFIG['start_index']}")
print(f"   • Índice final: {CONFIG['start_index'] + CONFIG['num_samples']}")
print(f"   • Features por texto: {CONFIG['lime_num_features']}")
print(f"   • Max tokens por texto: {CONFIG['max_tokens']}")
print(f"   • Arquivo de saída: {CONFIG['output_file']}")
print("="*70)

# ============================================================
# SETUP DO MODELO
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️  Usando dispositivo: {device}")

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
print(f"📦 Carregando modelo: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)
model.eval()

# ============================================================
# CARREGAR DATASET
# ============================================================
print(f"\n📚 Carregando dataset IMDB...")
end_index = CONFIG['start_index'] + CONFIG['num_samples']
dataset = load_dataset(
    "stanfordnlp/imdb", 
    split=f"train[{CONFIG['start_index']}:{end_index}]"
)
print(f"✅ Dataset carregado: {len(dataset)} amostras (índices {CONFIG['start_index']} a {end_index})")

# ============================================================
# FUNÇÃO DE PREDIÇÃO
# ============================================================
def predict_proba_lime(texts):
    """Função otimizada para LIME - batch processing"""
    if isinstance(texts, str):
        texts = [texts]
    
    # Limpa e valida textos (apenas remove espaços excessivos)
    texts_clean = []
    for text in texts:
        if isinstance(text, str) and text.strip():
            # Apenas normaliza espaços múltiplos, sem remover caracteres especiais
            clean = re.sub(r'\s+', ' ', text.strip())
            texts_clean.append(clean)
        else:
            texts_clean.append("empty text")
    
    if not texts_clean:
        return np.array([[0.5, 0.5]] * len(texts))
    
    try:
        inputs = tokenizer(
            texts_clean,
            padding=True,
            truncation=True,
            max_length=CONFIG['max_tokens'],  # Usa o valor configurado
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
        
        # Limpeza de memória
        del outputs, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return probs
        
    except Exception as e:
        print(f"⚠️  Erro na predição: {e}")
        return np.array([[0.5, 0.5]] * len(texts_clean))

# ============================================================
# ANÁLISE LIME EM LARGA ESCALA
# ============================================================
print(f"\n🔬 Iniciando análise LIME em {CONFIG['num_samples']} amostras...")
print(f"⏱️  Estimativa de tempo: ~{CONFIG['num_samples'] * 2 / 60:.1f} minutos\n")

# Estruturas para armazenar resultados agregados
word_scores_positive = defaultdict(list)  # Palavras que favorecem POSITIVO
word_scores_negative = defaultdict(list)  # Palavras que favorecem NEGATIVO
analysis_results = []

# Inicializa o explainer LIME
explainer = LimeTextExplainer(class_names=["NEGATIVE", "POSITIVE"])

# Barra de progresso
for idx in tqdm(range(len(dataset)), desc="Processando amostras"):
    try:
        # Pega o texto original sem limitação inicial
        text = dataset[idx]["text"]
        
        # Apenas remove espaços múltiplos (limpeza mínima)
        text_clean = re.sub(r'\s+', ' ', text.strip())
        
        if len(text_clean) < 20:  # Pula textos muito curtos
            continue
        
        # Predição do texto original
        pred = predict_proba_lime([text_clean])[0]
        predicted_class = np.argmax(pred)
        confidence = pred[predicted_class]
        
        # Gera explicação LIME
        exp = explainer.explain_instance(
            text_clean,
            predict_proba_lime,
            num_features=CONFIG['lime_num_features'],
            num_samples=CONFIG['lime_num_samples'],
            top_labels=1
        )
        
        # Extrai explicação para a classe predita
        explanation_list = exp.as_list(label=predicted_class)
        
        # Armazena resultados
        sample_result = {
            "index": CONFIG['start_index'] + idx,
            "predicted_class": int(predicted_class),
            "confidence": float(confidence),
            "text_preview": text_clean[:100],
            "explanations": []
        }
        
        # Processa cada palavra explicada
        for word_phrase, score in explanation_list:
            # Limpa apenas espaços extras, mantém pontuação e outros caracteres
            word_clean = word_phrase.strip().lower()
            
            # Aceita palavras válidas: >= 2 chars, não stopword, não número
            if len(word_clean) >= 2 and word_clean not in STOPWORDS and not word_clean.isdigit():
                sample_result["explanations"].append({
                    "word": word_clean,
                    "score": float(score)
            })
            
                # Classifica se a palavra favorece positivo ou negativo
                # Score positivo = favorece a classe predita
                # Score negativo = favorece a OUTRA classe
                if score > 0:
                    # Palavra favorece a classe que foi predita
                    if predicted_class == 1:  # POSITIVO
                        word_scores_positive[word_clean].append(score)
                    else:  # NEGATIVO
                        word_scores_negative[word_clean].append(score)
                else:
                    # Palavra favorece a classe OPOSTA à predita
                    if predicted_class == 1:  # Predição POSITIVA, mas palavra puxa pra NEGATIVA
                        word_scores_negative[word_clean].append(abs(score))
                    else:  # Predição NEGATIVA, mas palavra puxa pra POSITIVA
                        word_scores_positive[word_clean].append(abs(score))
        
        analysis_results.append(sample_result)
        
        # Mostra progresso detalhado
        if (idx + 1) % CONFIG['show_progress_every'] == 0:
            print(f"\n✓ Processadas {idx + 1}/{len(dataset)} amostras")
            print(f"  Palavras positivas únicas: {len(word_scores_positive)}")
            print(f"  Palavras negativas únicas: {len(word_scores_negative)}")
        
        # Limpeza de memória periódica
        if idx % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    except Exception as e:
        print(f"\n⚠️  Erro na amostra {idx}: {str(e)[:50]}...")
        continue

# ============================================================
# AGREGAÇÃO E RANKING
# ============================================================
print("\n\n📊 AGREGANDO RESULTADOS...")

def calculate_word_statistics(word_scores_dict):
    """Calcula estatísticas agregadas para cada palavra"""
    word_stats = {}
    for word, scores in word_scores_dict.items():
        word_stats[word] = {
            "mean_score": np.mean(scores),
            "total_score": np.sum(scores),
            "frequency": len(scores),
            "std": np.std(scores),
            "max": np.max(scores),
            "min": np.min(scores)
        }
    return word_stats

# Calcula estatísticas
positive_stats = calculate_word_statistics(word_scores_positive)
negative_stats = calculate_word_statistics(word_scores_negative)

# Cria rankings (por score total = impacto acumulado)
top_positive = sorted(
    positive_stats.items(),
    key=lambda x: x[1]['mean_score'] * np.sqrt(x[1]['frequency']),
    reverse=True
)

top_negative = sorted(
    negative_stats.items(),
    key=lambda x: x[1]['mean_score'] * np.sqrt(x[1]['frequency']),
    reverse=True
)

# ============================================================
# SALVAR RESULTADOS
# ============================================================
results_to_save = {
    "config": CONFIG,
    "metadata": {
        "total_samples_processed": len(analysis_results),
        "unique_positive_words": len(positive_stats),
        "unique_negative_words": len(negative_stats),
        "start_index": CONFIG['start_index'],
        "end_index": end_index
    },
    "top_50_positive_words": [
        {
            "rank": i + 1,
            "word": word,
            "mean_score": stats["mean_score"],
            "total_score": stats["total_score"],
            "frequency": stats["frequency"]
        }
        for i, (word, stats) in enumerate(top_positive[:50])
    ],
    "top_50_negative_words": [
        {
            "rank": i + 1,
            "word": word,
            "mean_score": stats["mean_score"],
            "total_score": stats["total_score"],
            "frequency": stats["frequency"]
        }
        for i, (word, stats) in enumerate(top_negative[:50])
    ],
    "all_positive_words": positive_stats,
    "all_negative_words": negative_stats,
    "sample_analyses": analysis_results[:20]  # Salva apenas 20 exemplos completos
}

with open(CONFIG['output_file'], 'w', encoding='utf-8') as f:
    json.dump(results_to_save, f, indent=2, ensure_ascii=False)

print(f"✅ Resultados salvos em: {CONFIG['output_file']}")

# ============================================================
# EXIBIÇÃO DOS RESULTADOS
# ============================================================
print("\n" + "="*70)
print(" 🏆 TOP 50 PALAVRAS MAIS POSITIVAS")
print("="*70)
print(f"{'Rank':<6} {'Palavra':<20} {'Score Médio':<15} {'Frequência':<12} {'Score Total':<12}")
print("-"*70)
for i, (word, stats) in enumerate(top_positive[:50], 1):
    print(f"{i:<6} {word:<20} {stats['mean_score']:>12.4f}   {stats['frequency']:>10}   {stats['total_score']:>10.2f}")

print("\n" + "="*70)
print(" 🏆 TOP 50 PALAVRAS MAIS NEGATIVAS")
print("="*70)
print(f"{'Rank':<6} {'Palavra':<20} {'Score Médio':<15} {'Frequência':<12} {'Score Total':<12}")
print("-"*70)
for i, (word, stats) in enumerate(top_negative[:50], 1):
    print(f"{i:<6} {word:<20} {stats['mean_score']:>12.4f}   {stats['frequency']:>10}   {stats['total_score']:>10.2f}")

print("\n" + "="*70)
print(" 📈 ESTATÍSTICAS GERAIS")
print("="*70)
print(f"Amostras processadas: {len(analysis_results)}")
print(f"Palavras positivas únicas: {len(positive_stats)}")
print(f"Palavras negativas únicas: {len(negative_stats)}")
print(f"Total de palavras únicas: {len(positive_stats) + len(negative_stats)}")
print("="*70)

print(f"\n💡 PRÓXIMOS PASSOS:")
print(f"   1. Para processar mais amostras, ajuste CONFIG['num_samples']")
print(f"   2. Para processar outro lote, ajuste CONFIG['start_index'] para {end_index}")
print(f"   3. Mude CONFIG['output_file'] para evitar sobrescrever resultados")
print(f"   4. Para agregar múltiplos batches, use o script de merge (veja documentação)")
print("\n✅ Análise concluída!")