🔍 Análise de Palavras Decisivas com LIME e DistilBERT

Este projeto identifica as palavras mais influentes na classificação de sentimentos em textos do dataset IMDB, usando DistilBERT fine-tuned e explicações locais com LIME.

🚀 Tecnologias Utilizadas

Python 🐍

PyTorch, Transformers (Hugging Face)

LIME para interpretabilidade

Datasets (Hugging Face)

NumPy, JSON, regex, tqdm

⚙️ Etapas Principais

Pré-processamento de textos do IMDB

Predição de sentimentos com DistilBERT

Explicação de decisões com LIME (palavras que favorecem POSITIVO ou NEGATIVO)

Agregação de estatísticas: score médio, frequência e impacto total por palavra

Salvamento dos resultados em JSON, incluindo top palavras e exemplos de análises

📊 Resultados

Ranking das 50 palavras mais positivas e 50 mais negativas

Estatísticas detalhadas de todas as palavras analisadas

Amostras explicadas individualmente pelo LIME
