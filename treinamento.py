import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
import shap
from lime.lime_text import LimeTextExplainer
import gc

# 1. Configuração do dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

# 2. Carregar modelo e tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)
model.eval()

# 3. Carregar dataset de exemplo (apenas 2 amostras para teste)
dataset = load_dataset("stanfordnlp/imdb", split="train[:2]")

# 4. Função de predição CORRIGIDA para LIME
def predict_proba_lime(texts):
    """Função específica para LIME - sempre retorna array 2D"""
    if isinstance(texts, str):
        texts = [texts]
    
    # Garante que é lista de strings
    texts_clean = []
    for text in texts:
        if isinstance(text, str) and text.strip():
            texts_clean.append(text[:512])  # Limita tamanho
        else:
            texts_clean.append("empty text")  # Fallback para textos vazios
    
    if not texts_clean:
        # Retorna probabilidades neutras se não há texto válido
        return np.array([[0.5, 0.5]] * len(texts))
    
    try:
        inputs = tokenizer(
            texts_clean,
            padding=True,
            truncation=True,
            max_length=128,
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
        print(f"Erro na predição: {e}")
        # Retorna probabilidades neutras em caso de erro
        return np.array([[0.5, 0.5]] * len(texts_clean))

# 5. Função de predição CORRIGIDA para SHAP
def predict_for_shap(texts):
    """Função específica para SHAP - retorna apenas uma dimensão"""
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, np.ndarray):
        texts = texts.tolist()
    
    # Filtra textos válidos
    valid_texts = []
    for text in texts:
        if isinstance(text, str) and text.strip():
            valid_texts.append(text[:256])  # Limita ainda mais para SHAP
        else:
            valid_texts.append("empty")
    
    try:
        probs = predict_proba_lime(valid_texts)
        # Retorna apenas a probabilidade da classe positiva
        return probs[:, 1]
    except Exception as e:
        print(f"Erro na predição SHAP: {e}")
        return np.array([0.5] * len(valid_texts))

# 6. Teste das predições
sample_texts = [dataset[i]["text"] for i in range(2)]
print("Testando predições...")
preds = predict_proba_lime(sample_texts)
print("\nPredições (probabilidades):")
print(preds)

# 7. Explicabilidade com LIME (versão mais robusta)
print("\nIniciando explicação LIME...")
try:
    explainer_lime = LimeTextExplainer(class_names=["NEGATIVE", "POSITIVE"])
    
    # Usa texto mais curto e limpo
    idx = 0
    test_text = sample_texts[idx][:300].strip()  # Limita e remove espaços extras
    
    # Teste rápido da função de predição
    test_pred = predict_proba_lime([test_text])
    print(f"Teste da predição: {test_pred}")
    
    if test_pred.shape[1] == 2:  # Verifica se tem 2 classes
        exp = explainer_lime.explain_instance(
            test_text, 
            predict_proba_lime, 
            num_features=5,
            num_samples=50,  # Reduzido ainda mais
            labels=[0, 1]  # Especifica explicitamente as classes
        )
        
        print("\nExplicação LIME para a primeira amostra:")
        
        # Tenta diferentes formas de extrair a explicação
        try:
            # Tenta pegar explicação para classe 0 (negativa)
            if 0 in exp.local_exp:
                explanation_list = exp.as_list(label=0)
                print("Características que influenciam para NEGATIVO:")
                for word, score in explanation_list:
                    print(f"  '{word}': {score:.4f}")
        except:
            pass
            
        try:
            # Tenta pegar explicação para classe 1 (positiva)  
            if 1 in exp.local_exp:
                explanation_list = exp.as_list(label=1)
                print("Características que influenciam para POSITIVO:")
                for word, score in explanation_list:
                    print(f"  '{word}': {score:.4f}")
        except:
            pass
        
        # Se nenhuma das anteriores funcionou, usa método manual
        if not any(label in exp.local_exp for label in [0, 1]):
            print("Usando método de extração alternativo...")
            # Pega todas as chaves disponíveis
            available_labels = list(exp.local_exp.keys())
            print(f"Classes disponíveis: {available_labels}")
            
            if available_labels:
                label_to_use = available_labels[0]
                explanation_list = exp.as_list(label=label_to_use)
                print(f"Explicação para classe {label_to_use}:")
                for word, score in explanation_list:
                    print(f"  '{word}': {score:.4f}")
        
        # Salva explicação em HTML
        try:
            exp.save_to_file('lime_explanation.html')
            print("Explicação LIME salva em 'lime_explanation.html'")
        except Exception as save_error:
            print(f"Não foi possível salvar o arquivo HTML: {save_error}")
    else:
        print("Erro: predição não retornou 2 classes")
        
except Exception as e:
    print(f"LIME oficial falhou: {str(e)[:100]}...")
    print("Usando implementação LIME alternativa...")
    
    # Implementação LIME manual robusta
    try:
        test_text = sample_texts[0][:300].strip()
        
        # Remove caracteres especiais para análise mais limpa
        import re
        clean_text = re.sub(r'[^\w\s]', ' ', test_text)
        words = [w for w in clean_text.split() if len(w) > 2][:20]  # Palavras > 2 chars, max 20
        
        print(f"Analisando {len(words)} palavras do texto...")
        
        original_pred = predict_proba_lime([test_text])[0]
        print(f"Predição original: Neg={original_pred[0]:.4f}, Pos={original_pred[1]:.4f}")
        
        word_importance_lime = []
        
        for i, word in enumerate(words):
            # Substitui a palavra por espaços
            temp_text = test_text.replace(word, " ")
            temp_text = re.sub(r'\s+', ' ', temp_text).strip()  # Remove espaços extras
            
            if len(temp_text) > 10:  # Garante que sobrou texto significativo
                temp_pred = predict_proba_lime([temp_text])[0]
                
                # Calcula impacto na classe negativa e positiva
                neg_impact = original_pred[0] - temp_pred[0]
                pos_impact = original_pred[1] - temp_pred[1]
                
                # Usa o impacto mais significativo
                main_impact = pos_impact if abs(pos_impact) > abs(neg_impact) else -neg_impact
                
                word_importance_lime.append((word, main_impact))
                
                if i < 3:  # Debug para as 3 primeiras palavras
                    print(f"  '{word}': {main_impact:.4f}")
        
        # Ordena por importância absoluta
        word_importance_lime.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print("\n🔍 LIME Alternativo - Top 5 palavras mais importantes:")
        for word, importance in word_importance_lime[:5]:
            direction = "➕ POSITIVA" if importance > 0 else "➖ NEGATIVA"
            confidence = abs(importance) * 100
            print(f"  '{word}': {importance:.4f} (influência {direction}, força: {confidence:.1f}%)")
    
    except Exception as alt_error:
        print(f"Erro na análise alternativa: {alt_error}")
        print("LIME não pôde ser executado com este modelo/texto.")

# 8. Explicabilidade com SHAP (versão mais simples)
print("\nIniciando explicação SHAP...")
try:
    # Texto muito curto para teste
    short_text = sample_texts[0][:150].strip()
    
    # Teste da função de predição para SHAP
    test_shap_pred = predict_for_shap([short_text, short_text])  # Testa com 2 inputs
    print(f"Teste SHAP - Input: 2 textos, Output: {len(test_shap_pred)} valores")
    
    if len(test_shap_pred) == 2:  # Verifica consistência
        # Usa uma abordagem mais simples - apenas palavras importantes
        words = short_text.split()[:20]  # Limita a 20 palavras
        
        # Calcula importância por palavra removendo uma de cada vez
        original_pred = predict_for_shap([short_text])[0]
        word_importance = []
        
        for i, word in enumerate(words):
            # Cria texto sem a palavra atual
            temp_words = words.copy()
            temp_words[i] = "[MASK]"  # Substitui por token especial
            temp_text = " ".join(temp_words)
            
            # Predição sem a palavra
            temp_pred = predict_for_shap([temp_text])[0]
            importance = original_pred - temp_pred
            word_importance.append((word, importance))
        
        # Ordena por importância
        word_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print("\nTop 5 palavras mais importantes (análise simplificada):")
        for word, importance in word_importance[:5]:
            direction = "positiva" if importance > 0 else "negativa"
            print(f"  '{word}': {importance:.4f} (influência {direction})")
    
    else:
        print(f"Erro de consistência: esperado 2 outputs, obtido {len(test_shap_pred)}")
        
except Exception as e:
    print(f"Erro no SHAP: {e}")
    import traceback
    traceback.print_exc()

print("\nAnálise concluída!")
print("\nResumo:")
print(f"- Texto analisado: '{sample_texts[0][:100]}...'")
print(f"- Predição do modelo: {preds[0][1]:.4f} (probabilidade positiva)")
print(f"- Classificação: {'POSITIVA' if preds[0][1] > 0.5 else 'NEGATIVA'}")

gc.collect()  # Limpeza final de memória