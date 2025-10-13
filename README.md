# pi-univesp-2025-2-s3-t20-machine-learning

# 💡 Predict Cost API

API para prever o **custo estimado de produtos** com base em características como categoria, preço unitário, valor por cento, pedido mínimo e quantidade solicitada. A API utiliza um modelo de machine learning treinado, com opção de atualização (retraining) via rota.

---

## 🚀 Funcionalidades

- 🔍 `/model/predict`: Gera uma estimativa de custo total com base nos dados fornecidos.
- 🔄 `/model/update`: Atualiza o modelo com novos dados de treinamento.

---

## 📦 Entradas da rota predict

| Campo           | Tipo     | Descrição                                                                 |
|----------------|----------|--------------------------------------------------------------------------|
| `produto`       | string   | Nome ou identificador do produto                                         |
| `valor_unitario`| float    | Preço por unidade                                                         |
| `quantidade`    | integer  | Quantidade solicitada pelo cliente                                       |
| `resultado_total`    | float  | Resultado total da venda                                       |
| ...             | ...      | Outros campos relevantes que são preenchidos baseado em dados conhecidos apartir do campo produto e outros campos compostos                   |

## ⚙️ Configurações do arquivo env para integrar a api ao database:

|Campo|Descrição|
|-----|---------|
|`driver`| SQL usado juntamente com o seu driver|
|`user`| Username para acessar o database|
|`password`| Senha usada para autenticação |
|`host`| Servidor onde está localizado o database (no projeto foi usado solução cloud, mas pode ser local)|
|`port`| Porta para onde está aberta para o database, no servidor|
|`database`| Banco de dados |

