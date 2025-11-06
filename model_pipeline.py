import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

from rdt import HyperTransformer

from copulas.multivariate import GaussianMultivariate
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.preprocessing import StandardScaler

load_dotenv()

class ModelPipeline:
    
    def __init__(self):
        """
        Inicializa o pipeline carregando os dados de referência dos produtos
        uma única vez para evitar consultas repetidas ao banco de dados.
        """
        print("📦 Carregando dados de referência dos produtos em memória...")
        self.produtos_df = self._load_reference_data()
        print("✅ Dados de referência carregados com sucesso.")

    def _load_reference_data(self):
        """Carrega a tabela de produtos do banco de dados."""
        eng = self.get_engine()
        query = "SELECT produto, categoria, cento_preco, pedido_minimo FROM produtos"
        df = pd.read_sql(query, eng)
        df['produto'] = df['produto'].str.lower().str.strip()
        return df

    def get_basedata(self):
        
        eng = self.get_engine()
        
        query = ''' 
        
        SELECT 
        
        p.produto, p.categoria, p.cento_preco, p.pedido_minimo, 
        v.id_venda, (v.quantidade * v.preco_unitario) as receita_total , v.quantidade, v.preco_unitario
        
        FROM vendas v 
        
        INNER JOIN produtos p ON v.produto_id = p.id
        
        '''
        base = pd.read_sql(query, eng)
        base = base.dropna()
        base[base.select_dtypes(include='object').columns.to_list()] = base[base.select_dtypes(include='object').columns.to_list()].apply(lambda x: x.str.lower()).apply(lambda x: x.str.strip())
        
        tmp = pd.DataFrame()
        tmp[['produto','categoria']] = base[['produto','categoria']]
        tmp['pedido_centopreco'] = (base['pedido_minimo'] + base['cento_preco']) / (base['cento_preco'] + 1e-6)
        
        
        norm_noise = np.random.normal(loc=0.6, scale=0.05, size=base.shape[0])
        tmp['custo'] = base['cento_preco'] * norm_noise
        
        tmp['razao_preco_pedido_custo'] = base['preco_unitario'] / ((base['pedido_minimo'] * base['cento_preco']) + 1e-6)
        tmp['razao_receita_pedido_quantidade'] = base['receita_total'] / ((base['pedido_minimo'] * base['quantidade']) + 1e-6)
        tmp['variacao_centopreco_pedido'] = (base['pedido_minimo'] - base['cento_preco']) / (base['cento_preco'] + 1e-6)
        
        vendas_produto_categoria = base.groupby(['categoria', 'produto']).size().reset_index(name='vendas_produto')
        vendas_categoria = base.groupby('categoria').size().reset_index(name='vendas_categoria')

        probabilidades = pd.merge(vendas_produto_categoria, vendas_categoria, on='categoria')

        tmp = tmp.merge(probabilidades[['produto','categoria','vendas_produto','vendas_categoria']], 
                on=['produto','categoria'], 
                how='left')
        tmp['prob_produto_categoria'] = tmp['vendas_produto'] / tmp['vendas_categoria']

        tmp = tmp.drop(['produto','categoria'], axis=1)
        return tmp        
    def get_engine(self):
        
        engine = create_engine(f"{os.getenv('driver')}://{os.getenv('user')}:{os.getenv('password')}@{os.getenv('host')}:{os.getenv('port')}/{os.getenv('database')}")
        
        return engine
    
    def transform_basedata(self):
        
        base = self.get_basedata()
        target = base['custo']
        base = base.drop(['custo'],axis=1)
        
        transformer = HyperTransformer()
        transformer.detect_initial_config(base)
        
        return transformer.fit_transform(base), target, transformer
    
    def transform_syntheticdata(self, sample=False):
        
        base, target, transformer = self.transform_basedata()
        base['custo'] = target
        
        model = GaussianMultivariate()
        
        model.fit(base)
        
        return model.sample(sample) if sample else model.sample(base.shape[0]), transformer
    
    def create_model(self, sample=False):
        
        base, transformer = self.transform_syntheticdata(sample)
        
        model = RandomForestRegressor(random_state=42)
        
        x_train, x_test, y_train, y_test = train_test_split(base.drop(['custo'], axis=1), base['custo'], test_size=0.2 )
        
        
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        
        model.fit(x_train_scaled, y_train)
        
        pred = model.predict(x_test_scaled)
        
        print('MSE ',mean_squared_error(y_test, pred)) 
        print('MAE ',mean_absolute_error(y_test, pred)) 
        print('R2 ',r2_score(y_test, pred))
        
        importance = pd.DataFrame({'importance':np.round(model.feature_importances_,7), 'x':x_train.columns})
        print('importance\n\n',importance.sort_values(by='importance', ascending=False))
        return model, transformer
    
    
    def tune_model(self, sample=False, n_iter=20):
        """
        Faz RandomizedSearchCV para regressão e exibe métricas de avaliação.
        """
        
        base, transformer = self.transform_syntheticdata(sample)

        X = base.drop('custo', axis=1)
        y = base['custo']

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        
        model = RandomForestRegressor(random_state=42)

        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [5,10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
        }

        search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )

        search.fit(x_train_scaled, y_train)
        best_model = search.best_estimator_

        pred = best_model.predict(x_test_scaled)

        print("\n🔍 Melhores hiperparâmetros:")
        print(search.best_params_)

        print("\n📊 Métricas:")
        print("MSE ", mean_squared_error(y_test, pred))
        print("MAE ", mean_absolute_error(y_test, pred))
        print("R2 ", r2_score(y_test, pred))

        importance = pd.DataFrame({
            'importance': np.round(best_model.feature_importances_, 7),
            'x': x_train.columns
        })
        print('\n🎯 Importância das features:\n', importance.sort_values(by='importance', ascending=False))

        return best_model, scaler, transformer if transformer else None
    
    def select_data(self, data):
        
        if not all(data.columns.isin(['produto','quantidade','preco_unitario','receita_total'])):
            raise ValueError('invalid parameter data')
        
        # Usa o DataFrame de produtos pré-carregado em vez de consultar o banco
        base = self.produtos_df.copy()

        data['produto'] = data['produto'].str.lower().str.strip()
        
        merged = base.merge(data, how='right', on='produto')
        merged = merged.dropna(subset=['categoria', 'cento_preco', 'pedido_minimo'])
        
        merged['pedido_centopreco'] = (merged['pedido_minimo'] + merged['cento_preco']) / (merged['cento_preco'] + 1e-6)
        merged['razao_preco_pedido_custo'] = merged['preco_unitario'] / ((merged['pedido_minimo'] * merged['cento_preco']) + 1e-6)
        merged['razao_receita_pedido_quantidade'] = merged['receita_total'] / ((merged['pedido_minimo'] * merged['quantidade']) + 1e-6)
        merged['variacao_centopreco_pedido'] = (merged['pedido_minimo'] - merged['cento_preco']) / (merged['cento_preco'] + 1e-6)

        vendas_produto_categoria = merged.groupby(['categoria', 'produto']).size().reset_index(name='vendas_produto')
        vendas_categoria = merged.groupby('categoria').size().reset_index(name='vendas_categoria')

        probabilidades = pd.merge(vendas_produto_categoria, vendas_categoria, on='categoria')

        merged = merged.merge(probabilidades[['produto','categoria','vendas_produto','vendas_categoria']], 
                on=['produto','categoria'], 
                how='left')
        merged['prob_produto_categoria'] = merged['vendas_produto'] / merged['vendas_categoria']
        
        merged = merged.drop(['produto','categoria','cento_preco', 'pedido_minimo', 'quantidade','preco_unitario','receita_total'], axis=1)
        
        return merged
        
    
    def predict(self, model, scaler: StandardScaler, transformer: HyperTransformer, data: pd.DataFrame):
        
        if not isinstance(scaler, StandardScaler) or not isinstance(transformer, HyperTransformer):
            raise ValueError('invalid model, scaler or transformer  parameter')
        
        
        if not all(data.columns.isin(['produto','quantidade','preco_unitario','receita_total'])):
            raise ValueError('invalid parameter data')
        
        data_selected = self.select_data(data)
        transformed = transformer.transform(data_selected)
        data_scaled = scaler.transform(transformed)
        
        predicted = model.predict(data_scaled)
        
        final = transformer.reverse_transform(transformed)
        final['custo_sugerido'] = np.round(predicted,2)
        
        final = pd.concat([final, data.drop(['produto'],axis=1)] , axis=1)
        return final
    
    def save_model(self, model, transformer):
        
        if not isinstance(model, RandomForestRegressor) or not isinstance(transformer, HyperTransformer):
            raise ValueError('invalid model or transformer parameter')
        
        joblib.dump({'model':model, 'transformer':transformer}, 'model.pkl')
        
        
if __name__ == '__main__':
    
    t = ModelPipeline()
    model, scaler, transformer = t.tune_model(sample=5000)
    
    data = pd.DataFrame([{'produto':'Coxinha de Frango', 'quantidade': 75, 'preco_unitario':0.8, 'receita_total': 60.0}])
    
    result = t.predict(model, scaler, transformer, data)
        
    print(pd.concat([data,result], axis=1).to_dict())