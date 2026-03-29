# src/utils.py
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os

class Utils:
    @staticmethod
    def calculate_severity_score(symptoms, severity_dict):
        """Calculate total severity score"""
        total = 0
        for symptom in symptoms:
            total += severity_dict.get(symptom, 0)
        return total
    
    @staticmethod
    def format_recommendations_text(disease_info, symptoms, confidence):
        """Format recommendations as text"""
        text = "="*70 + "\n"
        text += "MEDICINE RECOMMENDATION SYSTEM REPORT\n"
        text += "="*70 + "\n\n"
        
        text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        text += "PATIENT SYMPTOMS:\n"
        text += "-"*70 + "\n"
        for i, symptom in enumerate(symptoms, 1):
            text += f"{i}. {symptom}\n"
        text += "\n"
        
        text += "PREDICTED DISEASE:\n"
        text += "-"*70 + "\n"
        text += f"Disease: {disease_info['name']}\n"
        text += f"Confidence: {confidence:.2%}\n\n"
        
        text += "DESCRIPTION:\n"
        text += "-"*70 + "\n"
        text += f"{disease_info['description']}\n\n"
        
        text += "RECOMMENDED MEDICATIONS:\n"
        text += "-"*70 + "\n"
        for i, med in enumerate(disease_info['medications'][:10], 1):
            text += f"{i}. {med}\n"
        text += "\n"
        
        text += "DIET RECOMMENDATIONS:\n"
        text += "-"*70 + "\n"
        for i, diet in enumerate(disease_info['diets'][:10], 1):
            text += f"{i}. {diet}\n"
        text += "\n"
        
        text += "PRECAUTIONS:\n"
        text += "-"*70 + "\n"
        for i, prec in enumerate(disease_info['precautions'][:10], 1):
            text += f"{i}. {prec}\n"
        text += "\n"
        
        text += "="*70 + "\n"
        text += "END OF REPORT\n"
        text += "="*70 + "\n"
        
        return text
    
    @staticmethod
    def export_results(results, filename='recommendation_results.json'):
        """Export results to JSON"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results exported to {filename}")
    
    @staticmethod
    def get_symptom_statistics(symptoms_df):
        """Get statistics about symptoms"""
        all_symptoms = []
        for col in ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']:
            symptoms = symptoms_df[col].dropna().tolist()
            all_symptoms.extend([s.strip() for s in symptoms if s != ''])
        
        freq = {}
        for symptom in all_symptoms:
            freq[symptom] = freq.get(symptom, 0) + 1
        
        return freq