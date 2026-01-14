import pandas as pd
import optuna
import joblib
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from xgboost import XGBRegressor

# Wyłącz logowanie Optuny
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- KONFIGURACJA ---
FILE_PATH = "dane_bez_cev.csv"
POSTFIX = "60"

def load_data(path=FILE_PATH):
    if not os.path.exists(path):
        print(f"❌ Plik {path} nie istnieje.")
        return None, None, None

    try:
        df = pd.read_csv(path, sep=';', decimal=',')
        if df.shape[1] < 2:
            df = pd.read_csv(path, sep=',', decimal='.')

        feature_columns = ['~ C', '~ Mn', '~ Si', '~ Al', '~ Cr', '~ Ni']
        target_columns = ['minYield', 'minTensileStrength', 'minElongation']
        
        # Fallback nazw
        if not set(feature_columns).issubset(df.columns):
            feature_columns = ['c', 'mn', 'si', 'al', 'cr', 'ni']
            target_columns = ['yield', 'tensile', 'elongation']

        all_cols = feature_columns + target_columns
        existing_cols = [c for c in all_cols if c in df.columns]
        
        for col in existing_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=existing_cols)
        
        print(f"✅ Załadowano {len(df)} wierszy.")
        
        X = df[feature_columns]
        return X, df, feature_columns

    except Exception as e:
        print(f"❌ Błąd danych: {e}")
        return None, None, None

def create_pipeline(params):
    """
    Tworzy Pipeline: Scaler + Model.
    Dzięki temu nie musimy ręcznie skalować danych przed predykcją.
    """
    params = params.copy()
    model_type = params.pop('regressor')
    
    model = None

    if model_type == 'XGBoost':
        model = XGBRegressor(n_jobs=-1, **params)
    elif model_type == 'RandomForest':
        model = RandomForestRegressor(n_jobs=-1, **params)
    elif model_type == 'SVR':
        model = SVR(**params)
    
    # Tworzymy potok: Najpierw skalowanie, potem model
    # Dzięki temu StandardScaler uczy się tylko na tym, co widzi w danej chwili 
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    return pipeline

def objective(trial, X, y):
    """
    Optuna szuka parametrów.
    Teraz testuje różne kernele SVR.
    """
    regressor_name = trial.suggest_categorical("regressor", ["XGBoost", "RandomForest", "SVR"])
    
    clean_params = {'regressor': regressor_name}

    if regressor_name == "XGBoost":
        clean_params.update({
            'n_estimators': trial.suggest_int("xgb_n_estimators", 50, 200),
            'learning_rate': trial.suggest_float("xgb_learning_rate", 0.01, 0.1),
            'max_depth': trial.suggest_int("xgb_max_depth", 1, 3), # Płytkie
            'subsample': trial.suggest_float("xgb_subsample", 0.6, 1.0),
            'colsample_bytree': trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
            'reg_alpha': trial.suggest_float("xgb_reg_alpha", 0.1, 10.0, log=True),
            'reg_lambda': trial.suggest_float("xgb_reg_lambda", 1.0, 20.0, log=True),
            'min_child_weight': trial.suggest_int("xgb_min_child_weight", 2, 8) 
        })

    elif regressor_name == "RandomForest":
        clean_params.update({
            'n_estimators': trial.suggest_int("rf_n_estimators", 50, 200),
            'max_depth': trial.suggest_int("rf_max_depth", 2, 5),
            'min_samples_split': trial.suggest_int("rf_min_samples_split", 4, 15),
            'min_samples_leaf': trial.suggest_int("rf_min_samples_leaf", 2, 8),
            'max_samples': trial.suggest_float("rf_max_samples", 0.5, 0.8)
        })
        
    elif regressor_name == "SVR":
        # Rozbudowana sekcja SVR - testujemy różne jądra (kernels)
        kernel = trial.suggest_categorical("svr_kernel", ["linear", "poly", "rbf", "sigmoid"])
        clean_params['kernel'] = kernel
        
        # C jest wspólne dla wszystkich
        clean_params['C'] = trial.suggest_float("svr_C", 0.1, 100.0, log=True)
        clean_params['epsilon'] = trial.suggest_float("svr_epsilon", 0.01, 0.5)
        
        if kernel == "poly":
            #  Przy małych danych max 3!
            clean_params['degree'] = trial.suggest_int("svr_degree", 1, 3)
            clean_params['gamma'] = 'scale'
            clean_params['coef0'] = trial.suggest_float("svr_coef0", 0.0, 1.0)
            
        elif kernel == "rbf" or kernel == "sigmoid":
            clean_params['gamma'] = trial.suggest_categorical("svr_gamma", ["scale", "auto"])
            if kernel == "sigmoid":
                clean_params['coef0'] = trial.suggest_float("svr_coef0", -1.0, 1.0)
                
        elif kernel == "linear":
            # Linear nie potrzebuje hiperparametru, ani gammy ani degree
            pass

    # Tworzymy pipeline (Scaler + Model)
    pipeline = create_pipeline(clean_params)
    
    # K-Fold CV na surowych danych (Pipeline sam sobie przeskaluje wewnątrz pętli)
    kf = KFold(n_splits=4, shuffle=True
    )
    scores = cross_val_score(pipeline, X, y, cv=kf, scoring='r2')
    
    return scores.mean()

def perform_bootstrap_validation(model_params, X, y, n_iterations=100):
    scores = []
    # X i y wchodzą tu jako DataFrame/Series lub numpy, ale RAW (nieprzeskalowane)
    X_arr = np.array(X)
    y_arr = np.array(y)
    n_samples = len(X)

    print(f"🔄 Bootstrap ({n_iterations})...", end="", flush=True)

    for i in range(n_iterations):
        indices = np.arange(n_samples)
        train_idx = resample(indices, replace=True, random_state=i)
        test_idx = np.setdiff1d(indices, train_idx)
        
        if len(test_idx) < 2: continue

        X_boot_train, y_boot_train = X_arr[train_idx], y_arr[train_idx]
        X_boot_test, y_boot_test = X_arr[test_idx], y_arr[test_idx]

        # Pipeline wewnątrz bootstrapa - uczy skaler na wylosowanej próbce
        pipeline = create_pipeline(model_params)
        pipeline.fit(X_boot_train, y_boot_train)
        
        score = r2_score(y_boot_test, pipeline.predict(X_boot_test))
        scores.append(score)
    
    print(" Gotowe.")
    return scores

def main():
    X_raw, df_full, feature_names = load_data()
    if X_raw is None: return

    target_map = {
        'yield': ['minYield', 'Yield Strength', 'yield'],
        'tensile': ['minTensileStrength', 'minTensileStrenght', 'Tensile Strength', 'tensile'],
        'elongation': ['minElongation', 'Elongation', 'elongation']
    }

    for key, candidates in target_map.items():
        col_name = next((c for c in candidates if c in df_full.columns), None)
        if not col_name: continue

        print(f"\n🧠 --- Cel: {col_name} (Dataset 60 + Pipeline) ---")
        y_target = df_full[col_name]

        # Train/Test na surowych danych
        X_train, X_test, y_train, y_test = train_test_split(X_raw, y_target, test_size=0.15)

        # Optuna (przekazujemy surowe dane, objective sam buduje pipeline ze skalerem)
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=60)
        
        best_raw = study.best_params
        model_type = best_raw['regressor']
        
        # Odtwarzanie parametrów (bardziej skomplikowane przez zagnieżdżenia SVR)
        final_params = {'regressor': model_type}
        
        for k, v in best_raw.items():
            if k == 'regressor': continue
            
            # Usuwamy prefixy
            parts = k.split("_")
            param_name = "_".join(parts[1:]) 
            
            if model_type == "SVR":
                if k.startswith("svr_"):
                    param_key = k.replace("svr_", "")
                    final_params[param_key] = v
            else:
                final_params[param_name] = v
        
        print(f"🏆 Wybrano model: {model_type}")
        if model_type == 'SVR':
            print(f"   Kernel: {final_params.get('kernel')}")

        # Walidacja Bootstrapowa na SUROWYCH danych 
        scores = perform_bootstrap_validation(final_params, X_raw, y_target, n_iterations=100)
        
        mean_r2 = np.mean(scores)
        std_r2 = np.std(scores)

        print(f"📊 Wyniki Bootstrap:")
        print(f"   🔹 Średni R2:      {mean_r2:.4f}")
        print(f"   🔸 Odchylenie Std: {std_r2:.4f}")
        
        # Zapis FINALNEGO PIPELINE'U (Model + Scaler)
        final_pipeline = create_pipeline(final_params)
        final_pipeline.fit(X_raw, y_target)
        
        filename = f"{key}_model_{POSTFIX}.pkl"
        joblib.dump(final_pipeline, filename)
        print(f"💾 Zapisano PIPELINE do {filename} (zawiera wbudowany scaler)")

if __name__ == "__main__":
    main()