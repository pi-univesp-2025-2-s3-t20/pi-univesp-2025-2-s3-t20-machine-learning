﻿# pi-univesp-2025-2-s3-t20-machine-learning

# 💡 Predict Cost API

API para prever o **custo estimado de produtos** utilizando um modelo de Machine Learning (`RandomForestRegressor`). A API foi otimizada para deploy em ambientes com recursos limitados, como o plano gratuito da Koyeb, desacoplando o processo de treinamento do processo de inferência (servir predições).

---

## ⚙️ Estrutura e Fluxo de Trabalho

O projeto foi reestruturado para ser mais eficiente e robusto, seguindo as melhores práticas de MLOps para deploy.

1.  **Treinamento Offline**: O treinamento do modelo não acontece mais durante a inicialização da API. Ele é executado localmente através do script `model_pipeline.py`.
    ```bash
    python model_pipeline.py
    ```
    Este comando gera o arquivo `dl_model.pkl`, que contém o modelo treinado, o `StandardScaler` e o `HyperTransformer`.

2.  **Modelo Versionado**: O arquivo `dl_model.pkl` é "commitado" e versionado diretamente no repositório Git. Isso garante que a API sempre tenha um modelo pronto para usar.

3.  **API Leve para Inferência**: A aplicação Flask (`main.py`), servida com Gunicorn, tem a única responsabilidade de carregar o `dl_model.pkl` em memória e usar os artefatos para servir predições rapidamente. Isso resulta em um baixo consumo de memória e um tempo de inicialização muito rápido.

---

##  Endpoints da API

- `POST /model/predict`: Recebe os dados de uma venda e retorna uma estimativa de custo. **Importante**: o custo estimado refere-se ao lote total de itens da venda, não ao custo unitário.
- `GET /health`: Endpoint de health check que retorna o status da API. Útil para monitoramento em plataformas de nuvem.
- `GET /model/update`: **Desativada em produção.** Retorna uma mensagem indicando que o retreinamento deve ser feito offline.

---

## 📦 Entradas para a rota `/model/predict` (JSON)

| Campo           | Tipo     | Descrição                                         | Exemplo     |
|-----------------|----------|---------------------------------------------------|-------------|
| `produto`       | `string` | Nome do produto (deve existir na base de dados)   | `"coxinha"` |
| `quantidade`    | `integer`| Quantidade de itens na venda                      | `100`       |
| `preco_unitario`| `float`  | Preço de venda de cada unidade                    | `0.8`       |
| `receita_total` | `float`  | Receita total obtida com a venda (quantidade * preco_unitario) | `80.0`      |

---

## 🏁 Como Executar

1.  **Clone o repositório:**
    ```bash
    git clone <url-do-repositorio>
    cd pi-univesp-2025-2-s3-t20-machine-learning
    ```

2.  **Crie e configure o ambiente:**
    Crie um arquivo `.env` na raiz do projeto (você pode copiar o `.env.example` se houver um) com as credenciais do seu banco de dados PostgreSQL.
    ```ini
    driver=postgresql+psycopg2
    user=myuser
    password=mypassword
    host=myhost.aivencloud.com
    port=12345
    database=mydatabase
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **(Opcional) Treine um novo modelo:**
    O repositório já contém um `dl_model.pkl` pré-treinado. Para gerar um novo, execute:
    ```bash
    python model_pipeline.py
    ```

5.  **Execute a API localmente:**
    ```bash
    python main.py
    ```
    A API estará disponível em `http://localhost:8000`.
