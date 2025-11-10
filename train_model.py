import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score
import joblib
import os
import sys

def train_malware_model():
    print("🚀 Iniciando entrenamiento del modelo de Random Forest...")
    print("=" * 60)
    
    # Verificar que el dataset existe
    dataset_path = 'datasets/TotalFeatures-ISCXFlowMeter.csv'
    if not os.path.exists(dataset_path):
        print(f"❌ ERROR: No se encuentra el dataset en {dataset_path}")
        print("   Por favor, asegúrate de que el archivo CSV esté en la carpeta 'datasets/'")
        return None
    
    # Cargar el dataset
    try:
        print("📁 Cargando dataset...")
        df = pd.read_csv(dataset_path)
        print(f"✅ Dataset cargado: {len(df)} registros, {len(df.columns)} características")
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        return None
    
    # Mostrar información del dataset
    print("\n📊 INFORMACIÓN DEL DATASET:")
    print(f"   • Total de registros: {len(df):,}")
    print(f"   • Número de características: {len(df.columns)}")
    print(f"   • Distribución de clases:")
    class_dist = df['calss'].value_counts()
    for clase, count in class_dist.items():
        percentage = (count / len(df)) * 100
        print(f"     - {clase}: {count:,} registros ({percentage:.2f}%)")
    
    # Preparar los datos
    print("\n🔧 PREPARANDO DATOS PARA ENTRENAMIENTO...")
    
    # Separar características y etiquetas
    X = df.drop('calss', axis=1)
    y = df['calss']
    
    print(f"   • Características (X): {X.shape}")
    print(f"   • Etiquetas (y): {y.shape}")
    
    # Codificar las etiquetas
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"   • Clases codificadas: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    # Limpiar datos: manejar valores infinitos y NaN
    print("   • Limpiando datos (NaN e infinitos)...")
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Verificar valores nulos
    null_count = X.isnull().sum().sum()
    if null_count > 0:
        print(f"   • Encontrados {null_count} valores nulos, llenando con medianas...")
        X = X.fillna(X.median())
    
    # Escalar características
    print("   • Escalando características...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dividir en train y test
    print("   • Dividiendo en conjuntos de entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"   ✓ Conjunto de entrenamiento: {X_train.shape[0]:,} muestras")
    print(f"   ✓ Conjunto de prueba: {X_test.shape[0]:,} muestras")
    
    # Entrenar el modelo
    print("\n🌲 ENTRENANDO RANDOM FOREST...")
    print("   Esto puede tomar varios minutos...")
    
    model = RandomForestClassifier(
        n_estimators=100,        # Número de árboles
        max_depth=20,            # Profundidad máxima
        min_samples_split=5,     # Mínimo muestras para dividir
        min_samples_leaf=2,      # Mínimo muestras en hoja
        random_state=42,         # Para reproducibilidad
        n_jobs=-1,              # Usar todos los cores
        verbose=1               # Mostrar progreso
    )
    
    model.fit(X_train, y_train)
    print("✅ Modelo entrenado exitosamente!")
    
    # Evaluar el modelo
    print("\n📈 EVALUANDO MODELO...")
    
    # Predicciones
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"   • Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   • F1 Score: {f1:.4f}")
    
    # Reporte de clasificación detallado
    print("\n📋 REPORTE DE CLASIFICACIÓN:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Importancia de características
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔝 TOP 10 CARACTERÍSTICAS MÁS IMPORTANTES:")
    print("=" * 50)
    for i, row in feature_importance.head(10).iterrows():
        print(f"   {i+1:2d}. {row['feature']:30} {row['importance']:.4f}")
    
    # Guardar el modelo y preprocesadores
    print("\n💾 GUARDANDO MODELO...")
    
    # Crear directorio si no existe
    os.makedirs('detector/ml_model', exist_ok=True)
    
    # Guardar componentes
    joblib.dump(model, 'detector/ml_model/random_forest_model.pkl')
    joblib.dump(scaler, 'detector/ml_model/scaler.pkl')
    joblib.dump(label_encoder, 'detector/ml_model/label_encoder.pkl')
    joblib.dump(list(X.columns), 'detector/ml_model/feature_names.pkl')
    
    print("✅ Todos los componentes guardados exitosamente!")
    
    # Información final
    print("\n🎉 ENTRENAMIENTO COMPLETADO!")
    print("=" * 50)
    print(f"   📁 Modelo: detector/ml_model/random_forest_model.pkl")
    print(f"   📁 Scaler: detector/ml_model/scaler.pkl") 
    print(f"   📁 Label encoder: detector/ml_model/label_encoder.pkl")
    print(f"   📁 Features: detector/ml_model/feature_names.pkl")
    print(f"   📊 Accuracy final: {accuracy*100:.2f}%")
    print(f"   📊 F1 Score final: {f1*100:.2f}%")
    print("\n🚀 Ahora puedes ejecutar el servidor Django:")
    print("   python manage.py runserver")
    
    return model, scaler, label_encoder

if __name__ == "__main__":
    print("🤖 ENTRENADOR DE MODELO DE DETECCIÓN DE MALWARE")
    print("=" * 60)
    
    # Verificar dependencias
    try:
        import sklearn
        import pandas
        import joblib
        print("✅ Todas las dependencias están instaladas")
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("   Instala las dependencias con: pip install scikit-learn pandas joblib")
        sys.exit(1)
    
    # Ejecutar entrenamiento
    train_malware_model()