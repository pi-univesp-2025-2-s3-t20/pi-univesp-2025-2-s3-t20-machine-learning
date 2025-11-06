import os
os.environ["KERAS_BACKEND"] = "jax"  # Set backend before importing keras-core

import keras
import numpy as np
import pandas as pd
import joblib
import pprint
import time

from keras import layers
from keras import optimizers, losses, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


from model_pipeline import ModelPipeline


class DL_pipeline(ModelPipeline):

    def create_nn_model(self, sample=False):
        print("🔄 Gerando dados sintéticos...")
        base, transformer = self.transform_syntheticdata(sample)

        X = base.drop('custo', axis=1)
        y = base['custo']

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("📏 Padronizando com StandardScaler...")
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        # Modelo com keras-core e backend JAX
        print("🧠 Treinando modelo com Keras + JAX...")
        model = keras.Sequential([
            layers.Input(shape=(x_train_scaled.shape[1],)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(
            optimizer=optimizers.Adam(1e-3),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanAbsoluteError()]
        )

        history = model.fit(
            x_train_scaled,
            y_train,
            validation_split=0.2,
            batch_size=32,
            epochs=100,
            verbose=1
        )

        # Avaliação
        pred = model.predict(x_test_scaled).flatten()

        print(f"\n✅ Avaliação final:")
        print(f"MSE: {mean_squared_error(y_test, pred):.2f}")
        print(f"MAE: {mean_absolute_error(y_test, pred):.2f}")
        print(f"R²: {r2_score(y_test, pred):.2f}")

        return model, scaler, transformer

    def predict(self, model, scaler, transformer, data: pd.DataFrame):
        if not all(data.columns.isin(['produto','quantidade','preco_unitario','receita_total'])):
            raise ValueError('invalid parameter data')

        data_selected = self.select_data(data)
        transformed = transformer.transform(data_selected)
        data_scaled = scaler.transform(transformed)

        predicted = model.predict(data_scaled).flatten()
        predicted = np.round(predicted, 2)

        final = transformer.reverse_transform(transformed)
        final['custo_sugerido'] = predicted

        final = pd.concat([final, data.drop(['produto'], axis=1)], axis=1)
        return final

    def save_model(self, model, scaler, transformer):
        """Salva o modelo, scaler e transformer em um único arquivo .pkl."""
        print(f"\n💾 Salvando artefatos em 'dl_model.pkl'...")
        joblib.dump({
            'model': model,
            'scaler': scaler,
            'transformer': transformer
        }, 'dl_model.pkl')
        print("✅ Artefatos salvos com sucesso!")
        


if __name__ == '__main__':
    """
    Este bloco serve para treinar o modelo e salvar os artefatos.
    Execute `python dl_model_pipeline.py` para gerar o arquivo 'dl_model.pkl'.
    """
    start_time = time.time()
    t = DL_pipeline()
    model, scaler, transformer = t.create_nn_model(sample=5000)
    t.save_model(model, scaler, transformer)
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n⏱️  Tempo total de treinamento e salvamento: {duration:.2f} segundos")
