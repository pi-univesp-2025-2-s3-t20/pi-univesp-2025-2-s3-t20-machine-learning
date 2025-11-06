from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os 
from dl_model_pipeline import DL_pipeline

app = Flask(__name__)

# --- OTIMIZAÇÃO PRINCIPAL ---
# Instancia o pipeline e carrega os artefatos do modelo UMA ÚNICA VEZ no escopo global.
# Isso evita que o banco de dados seja consultado e que os arquivos sejam lidos a cada requisição.
pipeline = DL_pipeline()
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
    
    try:
        pipeline = DL_pipeline()
        
        # O treinamento agora é feito executando dl_model_pipeline.py diretamente
        # Esta rota pode ser usada para disparar o retreinamento no servidor
        model, scaler, transformer = pipeline.create_nn_model(sample=10000) # Aumentando a amostra para o retreino
        pipeline.save_model(model, scaler, transformer) 
        
        return jsonify({'msg':'model updated'})
    except Exception as e:
        return jsonify({'msg':'An error has occurred during processing', 'error': str(e)})

def train_model_if_not_exists():
    """Verifica se o modelo existe e o treina caso não exista."""
    if not os.path.exists('dl_model.pkl'):
        print("⚠️  Modelo 'dl_model.pkl' não encontrado.")
        print("🚀  Iniciando treinamento do modelo antes de iniciar o servidor...")
        try:
            pipeline = DL_pipeline()
            model, scaler, transformer = pipeline.create_nn_model(sample=5000)
            pipeline.save_model(model, scaler, transformer)
            print("✅  Treinamento concluído e modelo salvo com sucesso!")
        except Exception as e:
            print(f"❌ Erro durante o treinamento inicial: {e}")
            # Decide se o app deve parar ou continuar sem o modelo
            # Neste caso, vamos parar para evitar que a API rode sem funcionalidade.
            raise SystemExit("Não foi possível treinar o modelo inicial. A aplicação será encerrada.")

# Executa o treinamento na inicialização, ANTES de o Gunicorn iniciar os workers.
# Isso garante que o modelo esteja pronto quando a aplicação começar a servir.
def initialize_app():
    """Função para preparar tudo que a aplicação precisa antes de iniciar."""
    global artifacts
    train_model_if_not_exists()
    print("🧠 Carregando artefatos do modelo ('dl_model.pkl') em memória...")
    artifacts = joblib.load('dl_model.pkl')
    print("✅ Artefatos do modelo carregados com sucesso.")

initialize_app()

if __name__ == '__main__':
    # Para desenvolvimento local, o Gunicorn não é usado.
    app.run(host='0.0.0.0', port=8000, debug=True)