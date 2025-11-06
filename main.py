from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os 
from dl_model_pipeline import DL_pipeline

app = Flask(__name__)

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
            
            pipeline = DL_pipeline()
            
            # Verifica se o modelo treinado existe
            if os.path.exists('dl_model.pkl'):
                artifacts = joblib.load('dl_model.pkl')
                model, scaler, transformer = artifacts['model'], artifacts['scaler'], artifacts['transformer']
            else:
                # Se o modelo não existir, retorna um erro claro. O treinamento deve ser feito separadamente.
                return jsonify({'msg': 'Erro: Modelo não encontrado. Execute o script de treinamento primeiro.'}), 500
                
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

if __name__ == '__main__':
    train_model_if_not_exists()
    app.run(debug=True, port=8000)