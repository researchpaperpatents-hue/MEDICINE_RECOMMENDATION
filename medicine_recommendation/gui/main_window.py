# gui/main_window.py
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predictor import DiseasePredictor
from src.utils import Utils

class MedicineRecommendationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Medicine Recommendation System")
        self.root.geometry("1400x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize predictor
        self.predictor = DiseasePredictor()
        
        # Check if model exists
        self.check_model_and_data()
        
        # Create GUI
        self.create_gui()
        
        # Load symptoms if model is ready
        if self.predictor.is_loaded:
            self.load_symptoms()
    
    def check_model_and_data(self):
        """Check if model and data are available"""
        from src.data_processor import DataProcessor
        
        model_file = 'models/disease_model.h5'
        
        if not os.path.exists(model_file):
            result = messagebox.askyesno(
                "Model Not Found",
                "Trained model not found. Do you want to train the model now?\n\n"
                "This may take a few minutes."
            )
            if result:
                self.train_model()
            else:
                messagebox.showwarning(
                    "Warning",
                    "Application will exit. Please run train.py first."
                )
                self.root.quit()
        else:
            # Load data processor and predictor
            self.data_processor = DataProcessor()
            self.data_processor.load_encoders()
            self.predictor.initialize(self.data_processor)
    
    def train_model(self):
        """Train the model"""
        from src.data_processor import DataProcessor
        from src.model_trainer import ModelTrainer
        
        # Create a loading window
        loading_window = tk.Toplevel(self.root)
        loading_window.title("Training Model")
        loading_window.geometry("400x150")
        loading_window.configure(bg='white')
        
        tk.Label(loading_window, text="Training Neural Network Model...", 
                font=('Arial', 12, 'bold'), bg='white').pack(pady=20)
        
        progress = ttk.Progressbar(loading_window, mode='indeterminate')
        progress.pack(pady=10, padx=20, fill='x')
        progress.start()
        
        tk.Label(loading_window, text="This may take a few minutes...", 
                font=('Arial', 10), bg='white').pack()
        
        loading_window.update()
        
        # Train in separate thread
        def train_thread():
            try:
                # Process data
                processor = DataProcessor()
                X, y = processor.load_and_process()
                processor.save_encoders()
                
                # Train model
                trainer = ModelTrainer()
                history = trainer.train(X, y, epochs=50, batch_size=32)
                trainer.save_model()
                trainer.plot_training_history()
                
                # Update GUI
                self.root.after(0, self.training_complete, loading_window, processor)
                
            except Exception as e:
                self.root.after(0, self.training_error, loading_window, str(e))
        
        threading.Thread(target=train_thread, daemon=True).start()
    
    def training_complete(self, loading_window, processor):
        """Handle training completion"""
        loading_window.destroy()
        self.data_processor = processor
        self.predictor.initialize(self.data_processor)
        self.load_symptoms()
        messagebox.showinfo("Success", "Model trained successfully!")
    
    def training_error(self, loading_window, error):
        """Handle training error"""
        loading_window.destroy()
        messagebox.showerror("Training Error", f"Error during training:\n{error}")
        self.root.quit()
    
    def create_gui(self):
        """Create the GUI interface"""
        # Title frame
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=100)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="🏥 Medicine Recommendation System",
            font=('Arial', 24, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(
            title_frame,
            text="AI-Powered Disease Prediction & Treatment Recommendations",
            font=('Arial', 11),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        subtitle_label.pack()
        
        # Create paned window
        paned = ttk.PanedWindow(self.root, orient='horizontal')
        paned.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel - Symptom selection
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        self.create_symptom_panel(left_frame)
        
        # Right panel - Results
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=2)
        self.create_results_panel(right_frame)
        
        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Ready",
            bd=1,
            relief='sunken',
            anchor='w',
            bg='#ecf0f1'
        )
        self.status_bar.pack(side='bottom', fill='x')
    
    def create_symptom_panel(self, parent):
        """Create symptom selection panel"""
        # Search frame
        search_frame = tk.Frame(parent, bg='white')
        search_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(
            search_frame,
            text="🔍 Search Symptoms:",
            bg='white',
            font=('Arial', 10, 'bold')
        ).pack(side='left', padx=5)
        
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.filter_symptoms)
        search_entry = tk.Entry(search_frame, textvariable=self.search_var, width=30)
        search_entry.pack(side='left', padx=5)
        
        # Symptom list frame
        list_frame = tk.Frame(parent, bg='white', relief='ridge', bd=2)
        list_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Listbox with scrollbar
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.symptom_listbox = tk.Listbox(
            list_frame,
            selectmode='multiple',
            yscrollcommand=scrollbar.set,
            font=('Arial', 10),
            height=20
        )
        self.symptom_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.symptom_listbox.yview)
        
        # Selected symptoms display
        selected_frame = tk.Frame(parent, bg='white', relief='ridge', bd=2)
        selected_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(
            selected_frame,
            text="📋 Selected Symptoms:",
            bg='white',
            font=('Arial', 10, 'bold')
        ).pack(anchor='w', padx=5, pady=5)
        
        self.selected_symptoms_text = tk.Text(
            selected_frame,
            height=4,
            width=30,
            wrap=tk.WORD,
            font=('Arial', 9)
        )
        self.selected_symptoms_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Button frame
        button_frame = tk.Frame(parent, bg='white')
        button_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(
            button_frame,
            text="🔮 Predict Disease",
            command=self.predict_disease,
            font=('Arial', 12, 'bold'),
            bg='#3498db',
            fg='white',
            padx=20,
            pady=8
        ).pack(side='left', padx=5)
        
        tk.Button(
            button_frame,
            text="🗑️ Clear Selection",
            command=self.clear_selection,
            font=('Arial', 11),
            bg='#e74c3c',
            fg='white',
            padx=20,
            pady=8
        ).pack(side='left', padx=5)
        
        # Bind selection event
        self.symptom_listbox.bind('<<ListboxSelect>>', self.update_selected_symptoms)
    
    def create_results_panel(self, parent):
        """Create results display panel"""
        # Notebook for tabs
        notebook = ttk.Notebook(parent)
        notebook.pack(fill='both', expand=True)
        
        # Disease Info Tab
        disease_frame = tk.Frame(notebook, bg='white')
        notebook.add(disease_frame, text="📊 Disease Information")
        self.disease_text = scrolledtext.ScrolledText(
            disease_frame,
            wrap=tk.WORD,
            font=('Arial', 10),
            bg='white'
        )
        self.disease_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Medications Tab
        med_frame = tk.Frame(notebook, bg='white')
        notebook.add(med_frame, text="💊 Medications")
        self.medications_text = scrolledtext.ScrolledText(
            med_frame,
            wrap=tk.WORD,
            font=('Arial', 10),
            bg='white'
        )
        self.medications_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Diet Tab
        diet_frame = tk.Frame(notebook, bg='white')
        notebook.add(diet_frame, text="🥗 Diet Recommendations")
        self.diet_text = scrolledtext.ScrolledText(
            diet_frame,
            wrap=tk.WORD,
            font=('Arial', 10),
            bg='white'
        )
        self.diet_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Precautions Tab
        precautions_frame = tk.Frame(notebook, bg='white')
        notebook.add(precautions_frame, text="⚠️ Precautions")
        self.precautions_text = scrolledtext.ScrolledText(
            precautions_frame,
            wrap=tk.WORD,
            font=('Arial', 10),
            bg='white'
        )
        self.precautions_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Export button frame
        export_frame = tk.Frame(parent, bg='white')
        export_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Button(
            export_frame,
            text="💾 Export Results",
            command=self.export_results,
            font=('Arial', 10),
            bg='#27ae60',
            fg='white',
            padx=15,
            pady=5
        ).pack(side='right', padx=5)
    
    def load_symptoms(self):
        """Load symptoms into listbox"""
        self.symptom_listbox.delete(0, tk.END)
        for symptom in self.data_processor.unique_symptoms:
            self.symptom_listbox.insert(tk.END, symptom)
    
    def filter_symptoms(self, *args):
        """Filter symptoms based on search text"""
        search_term = self.search_var.get().lower()
        self.symptom_listbox.delete(0, tk.END)
        
        for symptom in self.data_processor.unique_symptoms:
            if search_term in symptom.lower():
                self.symptom_listbox.insert(tk.END, symptom)
    
    def update_selected_symptoms(self, event):
        """Update selected symptoms display"""
        selected_indices = self.symptom_listbox.curselection()
        selected_symptoms = [self.symptom_listbox.get(i) for i in selected_indices]
        
        self.selected_symptoms_text.delete(1.0, tk.END)
        self.selected_symptoms_text.insert(1.0, '\n'.join(selected_symptoms))
    
    def predict_disease(self):
        """Predict disease based on selected symptoms"""
        selected_indices = self.symptom_listbox.curselection()
        
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select at least one symptom")
            return
        
        selected_symptoms = [self.symptom_listbox.get(i) for i in selected_indices]
        
        # Update status
        self.status_bar.config(text="🔮 Making prediction...")
        self.root.update()
        
        # Run prediction in separate thread
        threading.Thread(target=self._run_prediction, args=(selected_symptoms,), daemon=True).start()
    
    def _run_prediction(self, symptoms):
        """Run prediction in separate thread"""
        try:
            # Predict
            result = self.predictor.predict(symptoms)
            disease_info = self.predictor.get_recommendations(result['disease'])
            
            # Update GUI in main thread
            self.root.after(0, self.display_results, result, disease_info, symptoms)
            
            self.status_bar.config(
                text=f"✅ Prediction complete: {result['disease']} ({result['confidence']:.1%})"
            )
            
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Prediction Error", str(e))
            self.status_bar.config(text="❌ Prediction failed")
    
    def display_results(self, result, disease_info, symptoms):
        """Display prediction results"""
        # Clear all text widgets
        self.disease_text.delete(1.0, tk.END)
        self.medications_text.delete(1.0, tk.END)
        self.diet_text.delete(1.0, tk.END)
        self.precautions_text.delete(1.0, tk.END)
        
        # Disease information
        disease_text = f"""
{'='*60}
PREDICTED DISEASE
{'='*60}

Disease: {disease_info['name']}
Confidence: {result['confidence']:.2%}

{'='*60}
TOP PREDICTIONS
{'='*60}
"""
        for disease, conf in result['top_predictions'][:5]:
            disease_text += f"\n• {disease}: {conf:.2%}"
        
        disease_text += f"""

{'='*60}
PATIENT SYMPTOMS ({len(symptoms)})
{'='*60}
"""
        for i, symptom in enumerate(symptoms, 1):
            disease_text += f"\n{i}. {symptom}"
        
        disease_text += f"""

{'='*60}
DESCRIPTION
{'='*60}

{disease_info['description']}
"""
        
        self.disease_text.insert(1.0, disease_text)
        
        # Medications
        med_text = "RECOMMENDED MEDICATIONS\n" + "="*60 + "\n\n"
        for i, med in enumerate(disease_info['medications'], 1):
            med_text += f"{i}. {med}\n"
        self.medications_text.insert(1.0, med_text)
        
        # Diet
        diet_text = "DIET RECOMMENDATIONS\n" + "="*60 + "\n\n"
        for i, diet in enumerate(disease_info['diets'], 1):
            diet_text += f"{i}. {diet}\n"
        self.diet_text.insert(1.0, diet_text)
        
        # Precautions
        prec_text = "PRECAUTIONS\n" + "="*60 + "\n\n"
        for i, prec in enumerate(disease_info['precautions'], 1):
            prec_text += f"{i}. {prec}\n"
        self.precautions_text.insert(1.0, prec_text)
    
    def clear_selection(self):
        """Clear all selections"""
        self.symptom_listbox.selection_clear(0, tk.END)
        self.selected_symptoms_text.delete(1.0, tk.END)
        self.search_var.set("")
        self.filter_symptoms()
        
        # Clear results
        self.disease_text.delete(1.0, tk.END)
        self.medications_text.delete(1.0, tk.END)
        self.diet_text.delete(1.0, tk.END)
        self.precautions_text.delete(1.0, tk.END)
        
        self.status_bar.config(text="Selection cleared")
    
    def export_results(self):
        """Export results to file"""
        try:
            # Get current results
            disease_text = self.disease_text.get(1.0, tk.END).strip()
            
            if not disease_text or disease_text == "No prediction made yet":
                messagebox.showwarning("Warning", "No results to export")
                return
            
            # Ask for file location
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                initialfile=f"recommendation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
            
            if filename:
                with open(filename, 'w') as f:
                    f.write(disease_text)
                    f.write("\n\n" + "="*60 + "\n")
                    f.write(self.medications_text.get(1.0, tk.END))
                    f.write("\n" + "="*60 + "\n")
                    f.write(self.diet_text.get(1.0, tk.END))
                    f.write("\n" + "="*60 + "\n")
                    f.write(self.precautions_text.get(1.0, tk.END))
                
                messagebox.showinfo("Success", f"Results exported to:\n{filename}")
                
        except Exception as e:
            messagebox.showerror("Export Error", str(e))