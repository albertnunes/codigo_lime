🧠 Análise de Palavras Decisivas com LIME e DistilBERT no Dataset IMDB

Este projeto aplica Explainable AI (XAI) para identificar quais palavras influenciam mais as decisões de um modelo de linguagem treinado para classificação de sentimentos em textos de avaliações de filmes (dataset IMDB).

O sistema utiliza o DistilBERT fine-tuned no SST-2, aliado à técnica LIME (Local Interpretable Model-Agnostic Explanations), para explicar cada predição e gerar estatísticas agregadas sobre o vocabulário mais determinante para sentimentos positivos e negativos.

🚀 Tecnologias Utilizadas

Python 🐍

PyTorch – execução e inferência do modelo DistilBERT

Transformers (Hugging Face) – carregamento do modelo e tokenização

LIME – geração de explicações locais interpretáveis

Datasets (Hugging Face) – carregamento do dataset IMDB

NumPy / JSON / tqdm / regex – processamento, agregação e salvamento de resultados

⚙️ Etapas Principais

Configuração do Lote de Execução

Define número de amostras (num_samples), índice inicial (start_index) e parâmetros do LIME (num_features, num_samples).

Permite processar o dataset IMDB em batches para evitar sobrecarga de memória.

Carregamento do Modelo

Utiliza distilbert-base-uncased-finetuned-sst-2-english da Hugging Face.

Classifica textos como POSITIVO ou NEGATIVO.

Processamento das Amostras

Cada texto é limpo minimamente (remoção de espaços múltiplos).

O modelo prediz a classe e a confiança associada.

O LIME explica localmente a decisão do modelo, indicando o impacto de cada palavra.

Agregação e Estatísticas Globais

As palavras mais influentes são agregadas com estatísticas:

mean_score: impacto médio

frequency: número de ocorrências

total_score: influência acumulada

Calcula rankings globais de palavras positivas e negativas.

Exportação de Resultados

Gera um arquivo .json com:

Configurações utilizadas

Palavras mais influentes (Top 50 positivas e negativas)

Estatísticas detalhadas de todas as palavras

Amostras de explicações individuais

📊 Saídas do Sistema

Arquivo JSON (ex: lime_results_batch.json4) contendo:

top_50_positive_words

top_50_negative_words

Estatísticas completas das palavras

Amostras com explicações do LIME

Além disso, o script exibe no console:

As 50 palavras mais positivas e negativas com scores médios, frequências e impacto total.

Estatísticas gerais do processamento (número de amostras e palavras únicas analisadas).
