from flask import Flask, request, jsonify
import pandas as pd
import json
import joblib
import os 
from model_pipeline import ModelPipeline


app = Flask()



@app.route('/model/predict', methods=['POST'])
def predict():
    
    if request.method == 'POST':
        try:
            
            form = request.get_json()
        
            data = json.loads(form)
            data = pd.DataFrame([data])
            
            pipeline = ModelPipeline()
            
            if os.path.exists('model.pkl'):
                model, transformer = joblib.load('model.pkl')
            else:
                model = pipeline.create_model(sample=5000)
                
            return jsonify(pipeline.predict(model, transformer, data).to_dict(orient='records'))
        except ValueError or RuntimeError:
            return jsonify({'msg':'The provided parameters appear to be invalid'})
        
        
    return jsonify({'msg':'This endpoint only accepts POST requests'})

@app.route('/model/update')
def retrain():
    
    try:
        pipeline = ModelPipeline()
        
        model, transformer = pipeline.tune_model()
        
        pipeline.save_model(model, transformer)
        
        return jsonify({'msg':'model updated'})
    except:
        return jsonify({'msg':'An error has occurred during processing'})