import requests
import json
import joblib
import pandas as pd  # Pipeline wymaga DataFrame z nazwami kolumn
import numpy as np
import random
from datetime import datetime

# --- KONFIGURACJA SIECIOWA ---
ESP32_IP = "192.168.1.40"  # Ustaw
URL = f"http://{ESP32_IP}/update"

# --- NAZWY KOLUMN  ---
FEATURE_COLUMNS = ['c', 'mn', 'si', 'al', 'cr', 'ni']

def generate_random_composition():
    """Generuje losowy skład chemiczny stali."""
    # Losowanie w typowych zakresach
    c_val = round(random.uniform(0.10, 0.50), 3)
    mn_val = round(random.uniform(0.40, 1.60), 3)
    si_val = round(random.uniform(0.10, 0.60), 3)
    al_val = round(random.uniform(0.010, 0.060), 3)
    cr_val = round(random.uniform(0.05, 1.50), 3)
    ni_val = round(random.uniform(0.05, 1.50), 3)
    
    composition = {
        'c': c_val,
        'mn': mn_val,
        'si': si_val,
        'al': al_val,
        'cr': cr_val,
        'ni': ni_val
    }
    
    print("\nWylosowany skład chemiczny:")
    print(f"   C: {c_val}% | Mn: {mn_val}% | Si: {si_val}% | Al: {al_val}% | Cr: {cr_val}% | Ni: {ni_val}%")
    
    return composition

def predict_properties(composition_dict):
    """Dokonuje predykcji używając modeli Pipeline."""
    try:
        # 1. Konwersja na DataFrame (Pipeline wymaga nazw kolumn!)
        df_input = pd.DataFrame([composition_dict])

        # 2. Ładowanie modeli (Pipeline zawiera już skaler w środku)
        model_yield = joblib.load('yield_model_123.pkl')
        model_tensile = joblib.load('tensile_model_123.pkl')
        model_elong = joblib.load('elongation_model_123.pkl')

        print("Modele załadowane pomyślnie.")

        # 3. Predykcja 
        y_yield = model_yield.predict(df_input)[0]
        y_tensile = model_tensile.predict(df_input)[0]
        y_elong = model_elong.predict(df_input)[0]

        print(f"Wyniki predykcji:")
        print(f"   - Re (Yield): {y_yield:.2f} MPa")
        print(f"   - Rm (Tensile): {y_tensile:.2f} MPa")
        print(f"   - A (Elongation): {y_elong:.2f} %")

        return int(y_yield), int(y_tensile), float(y_elong)

    except FileNotFoundError as e:
        print(f"BŁĄD: Nie znaleziono modelu. {e}")
        return None, None, None
    except Exception as e:
        print(f"BŁĄD predykcji: {e}")
        return None, None, None

def send_to_esp32(yield_val, tensile_val, elong_val):
    """Wysyła dane do ESP32."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    payload = {
        "yield": yield_val,
        "tensile": tensile_val,
        "elongation": round(elong_val, 1),
        "timestamp": timestamp
    }

    print(f"\n Wysyłam dane do {URL}...")

    try:
        response = requests.post(URL, json=payload, timeout=5)
        if response.status_code == 200:
            print("Sukces! ESP32 potwierdziło odbiór.")
        else:
            print(f"Serwer zwrócił kod: {response.status_code}")
            print(f"Treść: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Błąd połączenia z ESP32: {e}")

if __name__ == "__main__":
    print("--- KLIENT ML (Dataset 312 + Pipeline) ---")
    
    # 1. Losuj dane
    sklad = generate_random_composition()
    
    # 2. Oblicz właściwości
    re, rm, a5 = predict_properties(sklad)
    
    # 3. Wyślij
    if re is not None:
        send_to_esp32(re, rm, a5)