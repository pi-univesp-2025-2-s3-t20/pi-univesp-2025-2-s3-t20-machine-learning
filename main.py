from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os 
from model_pipeline import ModelPipeline

app = Flask(__name__)

# --- OTIMIZAÇÃO PRINCIPAL ---
# Instancia o pipeline e carrega os artefatos do modelo UMA ÚNICA VEZ no escopo global.
# Isso evita que o banco de dados seja consultado e que os arquivos sejam lidos a cada requisição.
pipeline = ModelPipeline()
artifacts = None

@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint de health check para verificar se a API está no ar.
    Retorna um status 200 OK com uma mensagem simples.
    """
    return jsonify({"status": "ok", "message": "API is running"}), 200

@app.route('/model/predict', methods=['POST'])
def predict():
    
    if request.method == 'POST':
        try:
            
            form = request.get_json()
        
            data = pd.DataFrame([form])
            
            # Usa os artefatos pré-carregados
            if artifacts is None:
                return jsonify({'msg': 'Erro: Artefatos do modelo não foram carregados na inicialização.'}), 500
            model, scaler, transformer = artifacts['model'], artifacts['scaler'], artifacts['transformer']
                
            return jsonify(pipeline.predict(model, scaler, transformer, data).to_dict(orient='records'))
        except (ValueError, RuntimeError):
            return jsonify({'msg':'The provided parameters appear to be invalid'})
        except Exception as e:
            return jsonify({'msg':'An error has occurred during processing', 'error': str(e)})
        
    return jsonify({'msg':'This endpoint only accepts POST requests'})

@app.route('/model/update')
def retrain():
    
    # Esta rota é desativada em produção para evitar o consumo excessivo de recursos.
    # O retreinamento deve ser feito em um ambiente separado.
    return jsonify({'msg':'A rota de retreinamento está desativada neste ambiente.'}), 403

# Executa o treinamento na inicialização, ANTES de o Gunicorn iniciar os workers.
# Isso garante que o modelo esteja pronto quando a aplicação começar a servir.
def initialize_app():
    """Função para preparar tudo que a aplicação precisa antes de iniciar."""
    global artifacts
    if not os.path.exists('dl_model.pkl'):
        raise FileNotFoundError("Arquivo de modelo 'dl_model.pkl' não encontrado. Treine o modelo localmente e faça o commit.")

    print("🧠 Carregando artefatos do modelo ('dl_model.pkl') em memória...")
    artifacts = joblib.load('dl_model.pkl')
    print("✅ Artefatos do modelo carregados com sucesso.")

initialize_app()

if __name__ == '__main__':
    # Para desenvolvimento local, o Gunicorn não é usado.
    app.run(host='0.0.0.0', port=8000, debug=True)