import pandas as pd
import numpy as np
import os
from django.shortcuts import render

class DataAnalyzer:
    def __init__(self):
        self.df = None
        
    def load_dataset(self):
        """Cargar el dataset real"""
        try:
            dataset_path = 'datasets/TotalFeatures-ISCXFlowMeter.csv'
            if not os.path.exists(dataset_path):
                return False
                
            self.df = pd.read_csv(dataset_path)
            return True
            
        except Exception as e:
            print(f"Error cargando dataset: {e}")
            return False
    
    def get_analysis_data(self):
        """Obtener datos para el análisis"""
        if self.df is None:
            # Datos de ejemplo si no hay dataset real
            return {
                'stats': {
                    'data_length': 631955,
                    'feature_count': 80,
                    'class_distribution': {
                        'benign': 471597,
                        'asware': 155613, 
                        'GeneralMalware': 4745
                    }
                }
            }
        
        try:
            # Estadísticas reales
            stats = {
                'data_length': len(self.df),
                'feature_count': len(self.df.columns),
                'class_distribution': self.df["calss"].value_counts().to_dict()
            }
            
            return {'stats': stats}
            
        except Exception as e:
            print(f"Error en análisis: {e}")
            # Datos de ejemplo en caso de error
            return {
                'stats': {
                    'data_length': 631955,
                    'feature_count': 80,
                    'class_distribution': {
                        'benign': 471597,
                        'asware': 155613, 
                        'GeneralMalware': 4745
                    }
                }
            }

# Instancia global del analizador
analyzer = DataAnalyzer()

def index(request):
    """Página principal con análisis de datos"""
    data_loaded = analyzer.load_dataset()
    analysis_data = analyzer.get_analysis_data()
    
    context = {
        'real_data_loaded': data_loaded,
        'stats': analysis_data['stats']
    }
    
    return render(request, 'index.html', context)