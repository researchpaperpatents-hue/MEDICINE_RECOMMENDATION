# src/data_processor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import os

class DataProcessor:
    def __init__(self, data_path='data/'):
        self.data_path = data_path
        self.symptom_encoder = None
        self.disease_encoder = None
        self.unique_symptoms = None
        self.unique_diseases = None
        
    def load_and_process(self):
        """Load and process all data files"""
        print("Loading data files...")
        
        # Load symptom-disease mapping
        symptoms_df = pd.read_csv(os.path.join(self.data_path, 'symtoms_df.csv'))
        
        # Load other data files
        descriptions_df = pd.read_csv(os.path.join(self.data_path, 'description.csv'))
        medications_df = pd.read_csv(os.path.join(self.data_path, 'medications.csv'))
        diets_df = pd.read_csv(os.path.join(self.data_path, 'diets.csv'))
        precautions_df = pd.read_csv(os.path.join(self.data_path, 'precautions_df.csv'))
        severity_df = pd.read_csv(os.path.join(self.data_path, 'Symptom-severity.csv'))
        
        # Create dictionaries
        self.descriptions = dict(zip(descriptions_df['Disease'], descriptions_df['Description']))
        self.medications = self._parse_list_column(medications_df, 'Medication')
        self.diets = self._parse_list_column(diets_df, 'Diet')
        self.precautions = self._parse_precautions(precautions_df)
        self.severity = dict(zip(severity_df['Symptom'], severity_df['weight']))
        
        # Extract unique symptoms
        all_symptoms = set()
        for col in ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']:
            symptoms = symptoms_df[col].dropna().unique()
            all_symptoms.update([s.strip() for s in symptoms if s != ''])
        
        self.unique_symptoms = sorted(all_symptoms)
        self.unique_diseases = symptoms_df['Disease'].unique()
        
        # Create encoders
        self.symptom_encoder = {symptom: idx for idx, symptom in enumerate(self.unique_symptoms)}
        self.disease_encoder = {disease: idx for idx, disease in enumerate(self.unique_diseases)}
        self.idx_to_disease = {idx: disease for disease, idx in self.disease_encoder.items()}
        
        # Create training data
        X, y = self._create_training_data(symptoms_df)
        
        print(f"Data processed successfully!")
        print(f"  - Symptoms: {len(self.unique_symptoms)}")
        print(f"  - Diseases: {len(self.unique_diseases)}")
        print(f"  - Training samples: {len(X)}")
        
        return X, y
    
    def _parse_list_column(self, df, column):
        """Parse list-like column data"""
        result = {}
        for _, row in df.iterrows():
            try:
                items = eval(row[column])
                result[row['Disease']] = items
            except:
                result[row['Disease']] = [row[column]]
        return result
    
    def _parse_precautions(self, df):
        """Parse precautions data"""
        result = {}
        for _, row in df.iterrows():
            precautions = []
            for i in range(1, 5):
                prec = row.get(f'Precaution_{i}', '')
                if pd.notna(prec) and prec != '':
                    precautions.append(prec)
            result[row['Disease']] = precautions
        return result
    
    def _create_training_data(self, symptoms_df):
        """Create feature vectors and labels for training"""
        X = []
        y = []
        
        for _, row in symptoms_df.iterrows():
            disease = row['Disease']
            symptom_vector = np.zeros(len(self.unique_symptoms))
            
            for col in ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']:
                symptom = row[col]
                if pd.notna(symptom) and symptom != '':
                    symptom = symptom.strip()
                    if symptom in self.symptom_encoder:
                        symptom_vector[self.symptom_encoder[symptom]] = 1
            
            X.append(symptom_vector)
            y.append(self.disease_encoder[disease])
        
        return np.array(X), np.array(y)
    
    def save_encoders(self, path='models/'):
        """Save encoders to disk"""
        import os
        os.makedirs(path, exist_ok=True)
        
        with open(os.path.join(path, 'symptom_encoder.pkl'), 'wb') as f:
            pickle.dump(self.symptom_encoder, f)
        
        with open(os.path.join(path, 'disease_encoder.pkl'), 'wb') as f:
            pickle.dump(self.disease_encoder, f)
        
        with open(os.path.join(path, 'idx_to_disease.pkl'), 'wb') as f:
            pickle.dump(self.idx_to_disease, f)
        
        with open(os.path.join(path, 'unique_symptoms.pkl'), 'wb') as f:
            pickle.dump(self.unique_symptoms, f)
        
        print(f"Encoders saved to {path}")
    
    def load_encoders(self, path='models/'):
        """Load encoders from disk"""
        import os
        
        with open(os.path.join(path, 'symptom_encoder.pkl'), 'rb') as f:
            self.symptom_encoder = pickle.load(f)
        
        with open(os.path.join(path, 'disease_encoder.pkl'), 'rb') as f:
            self.disease_encoder = pickle.load(f)
        
        with open(os.path.join(path, 'idx_to_disease.pkl'), 'rb') as f:
            self.idx_to_disease = pickle.load(f)
        
        with open(os.path.join(path, 'unique_symptoms.pkl'), 'rb') as f:
            self.unique_symptoms = pickle.load(f)
        
        print(f"Encoders loaded from {path}")
    
    def get_symptom_vector(self, symptoms):
        """Convert symptoms list to feature vector"""
        feature_vector = np.zeros(len(self.unique_symptoms))
        for symptom in symptoms:
            if symptom in self.symptom_encoder:
                feature_vector[self.symptom_encoder[symptom]] = 1
        return feature_vector
    
    def get_disease_info(self, disease):
        """Get all information about a disease"""
        return {
            'name': disease,
            'description': self.descriptions.get(disease, 'No description available'),
            'medications': self.medications.get(disease, []),
            'diets': self.diets.get(disease, []),
            'precautions': self.precautions.get(disease, [])
        }