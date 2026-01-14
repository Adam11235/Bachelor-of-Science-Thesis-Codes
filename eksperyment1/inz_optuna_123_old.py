import pandas as pd
import optuna
# import optuna.visualization as ov
import functools
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Wyłącz logowanie Optuny dla każdej próby w celu czystego logu
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_data(path=r"c:\Users\the30\OneDrive\Pulpit\inz\wetransfer_inzynierka_2025-12-21_1420\inzynierka\moj_dataset\dane_bez_cev.csv"):
    """Ładuje i przygotowuje dane."""
    try:
        df = pd.read_csv(path, sep=';', decimal=',')
        
        # Pierwotne kolumny i cechy
        feature_columns = ['~ C', '~ Cr','~ Ni','~ Mo','~ Mn','~ Al']
        target_columns = ['minYield', 'minTensileStrenght', 'minElongation']
        
        all_cols = feature_columns + target_columns
        
        for col in all_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                print(f"Ostrzeżenie: Kolumna '{col}' nie znaleziona w pliku.")
                
        # Usuwanie wierszy z brakującymi danymi w cechach lub celach
        df = df.dropna(subset=all_cols)
        
        # OStateczna lista cech
        final_feature_columns = feature_columns
        
        X = df[final_feature_columns]
        Y_df = df[target_columns]
        
        print(f"Dane załadowane pomyślnie. Liczba próbek: {len(X)}")
        return X, Y_df, target_columns

    except FileNotFoundError:
        print(f"Błąd: Plik '{path}' nie został znaleziony.")
        return None, None, None
    except KeyError as e:
        print(f"Błąd: Nie znaleziono kolumny w pliku CSV: {e}.")
        return None, None, None

def objective(trial, X_train, y_train, X_test, y_test):
    """Definicja funkcji celu dla Optuny."""
    
    # Skalowanie danych
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # 1. Wybór typu modelu
    model_type = trial.suggest_categorical('model_type', ['SVR', 'RandomForest', 'GradientBoosting'])
    
    model = None
    
    if model_type == 'SVR':
        # 1a. Wybór "wersji" SVR (kernela)
        svr_kernel = trial.suggest_categorical('svr_kernel', ['linear', 'rbf', 'poly'])
        
        if svr_kernel == 'linear':
            # Wersja 1: Liniowy SVR
            C = trial.suggest_float('C_linear', 1e-3, 1e3, log=True)
            model = SVR(kernel='linear', C=C)
            
        elif svr_kernel == 'rbf':
            # Wersja 2: SVR z kernelem RBF
            C = trial.suggest_float('C_rbf', 1e-3, 1e3, log=True)
            gamma = trial.suggest_float('gamma_rbf', 1e-4, 1e-1, log=True)
            model = SVR(kernel='rbf', C=C, gamma=gamma)
            
        elif svr_kernel == 'poly':
            # Wersja 3: SVR z kernelem wielomianowym
            C = trial.suggest_float('C_poly', 1e-3, 1e3, log=True)
            gamma = trial.suggest_float('gamma_poly', 1e-4, 1e-1, log=True)
            degree = trial.suggest_int('degree_poly', 2, 5)
            model = SVR(kernel='poly', C=C, gamma=gamma, degree=degree)
            
    elif model_type == 'RandomForest':
        # 2. Losowy Las Regresywny (Random Forest)
        n_estimators = trial.suggest_int('rf_n_estimators', 50, 500)
        max_depth = trial.suggest_int('rf_max_depth', 2, 32, log=True)
        min_samples_split = trial.suggest_int('rf_min_samples_split', 2, 10)
        model = RandomForestRegressor(n_estimators=n_estimators, 
                                      max_depth=max_depth,
                                      min_samples_split=min_samples_split,
                                     )
        
    elif model_type == 'GradientBoosting':
        # 3. Gradient Boosting
        n_estimators = trial.suggest_int('gb_n_estimators', 50, 500)
        learning_rate = trial.suggest_float('gb_learning_rate', 0.01, 0.3, log=True)
        max_depth = trial.suggest_int('gb_max_depth', 2, 10)
        model = GradientBoostingRegressor(n_estimators=n_estimators,
                                          learning_rate=learning_rate,
                                          max_depth=max_depth,
                                          )

    # Trenowanie i ocena modelu
    if model_type == 'SVR':
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        # RF i GB nie wymagają skalowania, ale użycie nieskalowanych danych 
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, preds)
    return mse

# --- Główna pętla wykonawcza ---
X, Y_df, target_columns = load_data()

if X is not None:
    # Ustawiamy liczbę prób dla Optuny
    N_TRIALS = 100

    for target_col in target_columns:
        print(f"\n{'='*60}")
        print(f"### Strojenie Optuną dla celu: {target_col} ###")
        print(f"{'='*60}")
        
        y = Y_df[target_col]
        
        # Podział na zbiory treningowe i testowe
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Użycie functools.partial do przekazania dodatkowych argumentów do 'objective'
        objective_with_data = functools.partial(objective, 
                                                X_train=X_train, 
                                                y_train=y_train, 
                                                X_test=X_test, 
                                                y_test=y_test)
        
        # Utworzenie i uruchomienie badania Optuny
        study = optuna.create_study(direction='minimize')
        study.optimize(objective_with_data, n_trials=N_TRIALS)
        
        print("\n--- Wyniki dla:", target_col, "---")
        print(f"Najlepsza wartość MSE: {study.best_value:.4f}")
        print("Najlepsze hiperparametry:")
        print(study.best_params)
        
        # Obliczenie R-squared dla najlepszego modelu
        best_params = study.best_params
        best_model = None
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if best_params['model_type'] == 'SVR':
            kernel = best_params['svr_kernel']
            if kernel == 'linear':
                best_model = SVR(kernel='linear', C=best_params['C_linear'])
            elif kernel == 'rbf':
                best_model = SVR(kernel='rbf', C=best_params['C_rbf'], gamma=best_params['gamma_rbf'])
            elif kernel == 'poly':
                best_model = SVR(kernel='poly', C=best_params['C_poly'], gamma=best_params['gamma_poly'], degree=best_params['degree_poly'])
        
        elif best_params['model_type'] == 'RandomForest':
            best_model = RandomForestRegressor(n_estimators=best_params['rf_n_estimators'],
                                               max_depth=best_params['rf_max_depth'],
                                               min_samples_split=best_params['rf_min_samples_split']
                                               )
        
        elif best_params['model_type'] == 'GradientBoosting':
            best_model = GradientBoostingRegressor(n_estimators=best_params['gb_n_estimators'],
                                                   learning_rate=best_params['gb_learning_rate'],
                                                   max_depth=best_params['gb_max_depth'],
                                                  )

        if best_model is not None:
            best_model.fit(X_train_scaled, y_train)
            final_preds = best_model.predict(X_test_scaled)
            r2 = r2_score(y_test, final_preds)
            print(f"R-squared (R2) dla najlepszego modelu: {r2:.4f}")
        else:
            print("Nie udało się utworzyć najlepszego modelu.")
    
    fig_history = ov.plot_optimization_history(study)
    fig_history.show()