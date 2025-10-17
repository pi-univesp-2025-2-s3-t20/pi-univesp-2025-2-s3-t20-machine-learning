from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os 
from dl_model_pipeline import DL_pipeline

app = Flask()

@app.route('/model/predict', methods=['POST'])
def predict():
    
    if request.method == 'POST':
        try:
            
            form = request.get_json()
        
            data = pd.DataFrame([form])
            
            pipeline = DL_pipeline()
            
            if os.path.exists('model.pkl'):
                model, scaler, transformer = joblib.load('model.pkl')
            else:
                model, scaler, transformer = pipeline.create_nn_model(sample=5000)
                pipeline.save_model(model, scaler, transformer)
                
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
        
        model, scaler, transformer = pipeline.create_nn_model(sample=5000)
        pipeline.save_model(model, scaler, transformer)
        
        return jsonify({'msg':'model updated'})
    except Exception as e:
        return jsonify({'msg':'An error has occurred during processing', 'error': str(e)})