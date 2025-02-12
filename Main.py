# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle
import tkinter as tk

# Importazione moduli con gestione errori
try:
    import RF_LG_DT, MNB_SVM, KMeans, KB, CSP_AnalysisPrenotation
    from CSP_AnalysisPrenotation import PrenotazioneGUI
    print("✅ Moduli importati correttamente.")
except ImportError as e:
    print(f"❌ Errore nell'importazione di un modulo: {e}")
    exit()

def main():
    print("📊 Preparazione dati in corso...")
    
    # Definizione delle feature
    feature = ["id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", 
               "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", 
               "concave_points_mean", "symmetry_mean", "fractal_dimension_mean", 
               "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", 
               "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", 
               "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", 
               "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", 
               "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"]
    
    feature_dummied = ["diagnosis"]  # Solo la colonna 'diagnosis' è categorica, quindi la trattiamo separatamente

    # Caricamento dataset
    try:
        dataset = pd.read_csv("data.csv", sep=",",header=0, names=feature, 
                              dtype={'id': np.int32, 'diagnosis': object, 'radius_mean': np.float64, 
                                     'texture_mean': np.float64, 'perimeter_mean': np.float64, 
                                     'area_mean': np.float64, 'smoothness_mean': np.float64, 
                                     'compactness_mean': np.float64, 'concavity_mean': np.float64, 
                                     'concave_points_mean': np.float64, 'symmetry_mean': np.float64, 
                                     'fractal_dimension_mean': np.float64, 'radius_se': np.float64, 
                                     'texture_se': np.float64, 'perimeter_se': np.float64, 
                                     'area_se': np.float64, 'smoothness_se': np.float64, 
                                     'compactness_se': np.float64, 'concavity_se': np.float64, 
                                     'concave_points_se': np.float64, 'symmetry_se': np.float64, 
                                     'fractal_dimension_se': np.float64, 'radius_worst': np.float64, 
                                     'texture_worst': np.float64, 'perimeter_worst': np.float64, 
                                     'area_worst': np.float64, 'smoothness_worst': np.float64, 
                                     'compactness_worst': np.float64, 'concavity_worst': np.float64, 
                                     'concave_points_worst': np.float64, 'symmetry_worst': np.float64, 
                                     'fractal_dimension_worst': np.float64})
        print("✅ Dataset caricato correttamente.")
    except FileNotFoundError:
        print("❌ Errore: File 'Breast_Cancer.csv' non trovato!")
        exit()

    # Conversione delle variabili categoriche in dummy variables
    data_dummies = pd.get_dummies(dataset, columns=feature_dummied)
    data_dummies = data_dummies.drop(["id"], axis=1)  # Rimuoviamo la colonna 'id', che non è utile per il modello

    # Feature e target
    x = data_dummies.drop(["diagnosis_B", "diagnosis_M"], axis=1)  # 'diagnosis_B' e 'diagnosis_M' sono le classi target
    y = dataset["diagnosis"].map({"B": 0, "M": 1})  # 0 per benigno, 1 per maligno

    print("✅ Preparazione dati completata.\n")

  # Classificazione non supervisionata
    print("\n🔹 ALGORITMO: K-Means")
    KMeans.KMEANS(x, y)

    # Classificazione supervisionata
    print("\n🔹 ALGORITMO: Random Forest")
    RF_LG_DT.RF(x, y)

      # Classificazione supervisionata
    print("\n🔹 ALGORITMO: DecisionTree")
    RF_LG_DT.DT(x, y)

     # Classificazione supervisionata
    print("\n🔹 ALGORITMO: Logistic Regression")
    RF_LG_DT.LR(x, y)

    print("\n🔹 ALGORITMI comparati Logistic Regression, DecisionTree, Random Forest")
    RF_LG_DT.plot_all_learning_curves(x, y)
    RF_LG_DT.plot_comparison_metrics(x, y)
    
    print("\n🔹 ALGORITMO: Multinomial Naive Bayes")
    MNB_SVM.NaiveBayes(x, y)

    print("\n🔹 ALGORITMO: Support Vector Machine")
    MNB_SVM.SVM(x, y)

    print("\n🔹 ALGORITMI SVM e MNB comparati")
    MNB_SVM.plot_comparison_learning_curves(x, y)
    MNB_SVM.plot_comparison_metrics(x, y)

    print("\n🔹 KB")
    KB.run_queries_and_display()
 # Prenotazione visita 
    print("\n📅 Prenotazione visita")
    root = tk.Tk()  # Crea la finestra principale Tkinter
    app = CSP_AnalysisPrenotation.PrenotazioneGUI(root)  # Passa root al costruttore di PrenotazioneGUI
    root.mainloop()

# Avvio del programma
if __name__ == "__main__":
    main()
   
