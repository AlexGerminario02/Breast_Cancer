# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sn
import pickle
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split,learning_curve, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

def NaiveBayes(X1, y1):
    # Suddivisione in set di addestramento e test
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, train_size=0.75, random_state=13)

    # Ottimizzazione dei parametri tramite GridSearch (combinazioni di alpha)
    param_grid = {'alpha': [0.01, 0.1, 0.5, 1, 2, 5, 10]}
    grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X1_train, y1_train)
    best_alpha = grid_search.best_params_['alpha']
    
    print(f"\nMiglior alpha per Naive Bayes: {best_alpha}")

    # Creazione e addestramento del modello con l'alpha ottimizzato
    clf_nb = MultinomialNB(alpha=best_alpha)
    clf_nb.fit(X1_train, y1_train)

    # Predizioni e accuratezza
    prediction_nb = clf_nb.predict(X1_test)
    accuracy_nb = accuracy_score(prediction_nb, y1_test)

    print("\n--- Performance Summary Naive Bayes ---")
    print(f"Accuracy: {accuracy_nb:.3f}")
    print("\n--- Classification Report ---")
    print(classification_report(y1_test, prediction_nb))

    # Calcola e visualizza la Curva ROC per Naive Bayes
    probs_nb = clf_nb.predict_proba(X1_test)[:, 1]  # Probabilità positive
    fpr_nb, tpr_nb, _ = roc_curve(y1_test, probs_nb)
    auc_nb = roc_auc_score(y1_test, probs_nb)
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
    plt.plot(fpr_nb, tpr_nb, marker='.', label='Naive Bayes ROC Curve (AUC = {:.2f})'.format(auc_nb))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Naive Bayes ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    plt.close()
    pass

    # Confusion Matrix per Naive Bayes
    confusion_Matrix_nb = confusion_matrix(y1_test, prediction_nb)
    print("\n--- Matrice di Confusione per Naive Bayes ---")
    print(confusion_Matrix_nb)

    # Visualizzazione Confusion Matrix come Heatmap per Naive Bayes
    df_cm_nb = pd.DataFrame(confusion_Matrix_nb, index=['Classe 0', 'Classe 1'], columns=['Classe 0', 'Classe 1'])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm_nb, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'size': 16})
    plt.title('Naive Bayes Confusion Matrix')
    plt.show()
    plt.close()
    pass
    # Precision-Recall Curve per Naive Bayes
    average_precision_nb = average_precision_score(y1_test, prediction_nb)
    precision_nb, recall_nb, _ = precision_recall_curve(y1_test, probs_nb)  # Usare probabilità

    plt.figure(figsize=(8, 6))
    plt.step(recall_nb, precision_nb, color='b', alpha=0.2, where='post')
    plt.fill_between(recall_nb, precision_nb, alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Naive Bayes Precision-Recall Curve: AP={average_precision_nb:.2f}')
    plt.show()
    plt.close()
    pass


def SVM(X1, y1):
    # Suddivisione del dataset in train e test (75% train, 25% test)
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, train_size=0.75, random_state=13)

    # Scaling delle features (molto importante per SVM)
    scaler = StandardScaler()
    X1_train_scaled = scaler.fit_transform(X1_train)
    X1_test_scaled = scaler.transform(X1_test)

    # Ottimizzazione dei parametri tramite GridSearchCV (C e gamma)
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 1],
        'kernel': ['linear', 'rbf', 'poly']  # Proviamo diversi kernel
    }
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X1_train_scaled, y1_train)
    best_params = grid_search.best_params_

    print(f"\nMigliori parametri trovati per SVM: {best_params}")

    # Creazione del modello SVM con i migliori parametri
    clf_svm = svm.SVC(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'])
    clf_svm.fit(X1_train_scaled, y1_train)

    # Predizioni
    prediction_svm = clf_svm.predict(X1_test_scaled)
    accuracy_svm = accuracy_score(y1_test, prediction_svm)

    # Report di classificazione
    print("\n--- Report di Classificazione per SVM ---")
    print(classification_report(y1_test, prediction_svm))

    # Calcola e visualizza la Curva ROC per SVM
    probs_svm = clf_svm.decision_function(X1_test_scaled)  # Usare decision_function per SVM
    fpr_svm, tpr_svm, _ = roc_curve(y1_test, probs_svm)
    auc_svm = roc_auc_score(y1_test, probs_svm)

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
    plt.plot(fpr_svm, tpr_svm, marker='.', label='SVM ROC Curve (AUC = {:.2f})'.format(auc_svm))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('SVM ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    plt.close()
    pass

    # Matrice di Confusione per SVM
    confusion_Matrix_svm = confusion_matrix(y1_test, prediction_svm)
    print("\n--- Matrice di Confusione per SVM ---")
    print(confusion_Matrix_svm)

    # Visualizzazione della Matrice di Confusione come Heatmap per SVM
    df_cm_svm = pd.DataFrame(confusion_Matrix_svm, index=['Classe 0', 'Classe 1'], columns=['Classe 0', 'Classe 1'])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm_svm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'size': 16})
    plt.title('SVM Confusion Matrix')
    plt.show()
    plt.close()
    pass

    # Precision-Recall Curve per SVM
    average_precision_svm = average_precision_score(y1_test, prediction_svm)
    precision_svm, recall_svm, _ = precision_recall_curve(y1_test, probs_svm)

    plt.figure(figsize=(8, 6))
    plt.step(recall_svm, precision_svm, color='g', alpha=0.2, where='post')
    plt.fill_between(recall_svm, precision_svm, alpha=0.2, color='g')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'SVM Precision-Recall Curve: AP={average_precision_svm:.2f}')
    plt.show()
    plt.close()
    pass


def plot_comparison_learning_curves(X, y):
    """
    Funzione per tracciare le curve di apprendimento di Naive Bayes e SVM nello stesso grafico.
    """
    # Definisci i modelli
    clf_nb = MultinomialNB()
    clf_svm = svm.SVC(kernel='rbf', C=1, gamma='scale')  # SVM con parametri di base

    # Calcola le curve di apprendimento
    train_sizes_nb, train_scores_nb, test_scores_nb = learning_curve(clf_nb, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
    train_sizes_svm, train_scores_svm, test_scores_svm = learning_curve(clf_svm, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

    # Calcolare le medie e le deviazioni standard
    train_mean_nb = np.mean(train_scores_nb, axis=1)
    train_std_nb = np.std(train_scores_nb, axis=1)
    test_mean_nb = np.mean(test_scores_nb, axis=1)
    test_std_nb = np.std(test_scores_nb, axis=1)

    train_mean_svm = np.mean(train_scores_svm, axis=1)
    train_std_svm = np.std(train_scores_svm, axis=1)
    test_mean_svm = np.mean(test_scores_svm, axis=1)
    test_std_svm = np.std(test_scores_svm, axis=1)

    # Traccia le curve di apprendimento
    plt.figure(figsize=(10, 6))

    # Naive Bayes
    plt.plot(train_sizes_nb, train_mean_nb, label="Naive Bayes Training Score", color="blue", marker="o")
    plt.plot(train_sizes_nb, test_mean_nb, label="Naive Bayes Cross-validation Score", color="red", marker="x")
    plt.fill_between(train_sizes_nb, train_mean_nb - train_std_nb, train_mean_nb + train_std_nb, color="blue", alpha=0.1)
    plt.fill_between(train_sizes_nb, test_mean_nb - test_std_nb, test_mean_nb + test_std_nb, color="red", alpha=0.1)

    # SVM
    plt.plot(train_sizes_svm, train_mean_svm, label="SVM Training Score", color="green", marker="o")
    plt.plot(train_sizes_svm, test_mean_svm, label="SVM Cross-validation Score", color="orange", marker="x")
    plt.fill_between(train_sizes_svm, train_mean_svm - train_std_svm, train_mean_svm + train_std_svm, color="green", alpha=0.1)
    plt.fill_between(train_sizes_svm, test_mean_svm - test_std_svm, test_mean_svm + test_std_svm, color="orange", alpha=0.1)

    # Personalizzazione del grafico
    plt.title('Confronto delle Curve di Apprendimento: Naive Bayes vs SVM')
    plt.xlabel("Numero di Campioni nel Training Set")
    plt.ylabel("Accuratezza")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
    plt.close()
    pass


def plot_comparison_metrics(X, y):
    """
    Funzione per tracciare le curve di comparazione tra le metriche Precision, Average Precision, Accuracy e Recall 
    di Naive Bayes e SVM.
    """
    # Definisci i modelli
    clf_nb = MultinomialNB()
    clf_svm = svm.SVC(kernel='rbf', C=1, gamma='scale')  # SVM con parametri di base

    # Calcola le curve di apprendimento per entrambi i modelli
    train_sizes_nb, train_scores_nb, test_scores_nb = learning_curve(clf_nb, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
    train_sizes_svm, train_scores_svm, test_scores_svm = learning_curve(clf_svm, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

    # Calcolare le medie e le deviazioni standard per i punteggi
    train_mean_nb = np.mean(train_scores_nb, axis=1)
    test_mean_nb = np.mean(test_scores_nb, axis=1)
    train_mean_svm = np.mean(train_scores_svm, axis=1)
    test_mean_svm = np.mean(test_scores_svm, axis=1)

    # Inizializzazione delle liste per le metriche
    precision_nb, recall_nb, avg_precision_nb, accuracy_nb = [], [], [], []
    precision_svm, recall_svm, avg_precision_svm, accuracy_svm = [], [], [], []

    for size in train_sizes_nb:
        # Divisione dei dati
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size, random_state=13)

        # Addestramento e predizione per Naive Bayes
        clf_nb.fit(X_train, y_train)
        y_pred_nb = clf_nb.predict(X_test)
        precision_nb.append(precision_score(y_test, y_pred_nb))
        recall_nb.append(recall_score(y_test, y_pred_nb))
        avg_precision_nb.append(average_precision_score(y_test, y_pred_nb))
        accuracy_nb.append(accuracy_score(y_test, y_pred_nb))

        # Addestramento e predizione per SVM
        clf_svm.fit(X_train, y_train)
        y_pred_svm = clf_svm.predict(X_test)
        precision_svm.append(precision_score(y_test, y_pred_svm))
        recall_svm.append(recall_score(y_test, y_pred_svm))
        avg_precision_svm.append(average_precision_score(y_test, y_pred_svm))
        accuracy_svm.append(accuracy_score(y_test, y_pred_svm))

    # Tracciare le metriche per Naive Bayes e SVM
    plt.figure(figsize=(10, 8))

    # Precision
    plt.subplot(2, 2, 1)
    plt.plot(train_sizes_nb, precision_nb, label="Naive Bayes", color="blue", marker="o")
    plt.plot(train_sizes_svm, precision_svm, label="SVM", color="green", marker="x")
    plt.xlabel("Training Set Size")
    plt.ylabel("Precision")
    plt.title("Precision Comparison")
    plt.legend()

    # Recall
    plt.subplot(2, 2, 2)
    plt.plot(train_sizes_nb, recall_nb, label="Naive Bayes", color="blue", marker="o")
    plt.plot(train_sizes_svm, recall_svm, label="SVM", color="green", marker="x")
    plt.xlabel("Training Set Size")
    plt.ylabel("Recall")
    plt.title("Recall Comparison")
    plt.legend()

    # Average Precision
    plt.subplot(2, 2, 3)
    plt.plot(train_sizes_nb, avg_precision_nb, label="Naive Bayes", color="blue", marker="o")
    plt.plot(train_sizes_svm, avg_precision_svm, label="SVM", color="green", marker="x")
    plt.xlabel("Training Set Size")
    plt.ylabel("Average Precision")
    plt.title("Average Precision Comparison")
    plt.legend()

    # Accuracy
    plt.subplot(2, 2, 4)
    plt.plot(train_sizes_nb, accuracy_nb, label="Naive Bayes", color="blue", marker="o")
    plt.plot(train_sizes_svm, accuracy_svm, label="SVM", color="green", marker="x")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison")
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.close()
    pass




# Carica il dataset (ad esempio, Breast Cancer dataset da sklearn)
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target

