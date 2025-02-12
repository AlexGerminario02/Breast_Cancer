# -*- coding: utf-8 -*-

import numpy as np, pandas as pd, seaborn as sn
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score , precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_comparison_metrics(X1, y1):
    """
    Confronta Random Forest, Decision Tree e Logistic Regression in base a:
    - Accuracy
    - F1-Score
    - Precision
    - Average Precision (AP)
    """
    # Dizionario per memorizzare le metriche
    metrics = {
        "Random Forest": {},
        "Decision Tree": {},
        "Logistic Regression": {}
    }

    # Modelli
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=20, random_state=0),
        "Decision Tree": DecisionTreeClassifier(random_state=0),
        "Logistic Regression": LogisticRegression(random_state=0, max_iter=50000, solver='liblinear')
    }

    # Standardizzazione per Logistic Regression
    scaler = StandardScaler()
    X1_scaled = scaler.fit_transform(X1)

    # Loop sui modelli
    for model_name, model in models.items():
        # Per Logistic Regression usiamo X1_scaled
        X1_data = X1_scaled if model_name == "Logistic Regression" else X1

        # Split dei dati
        X1_train, X1_test, y1_train, y1_test = train_test_split(X1_data, y1, train_size=0.7, random_state=13)

        # Training del modello
        model.fit(X1_train, y1_train)
        prediction = model.predict(X1_test)

        # Calcolo delle metriche
        accuracy = accuracy_score(y1_test, prediction)
        f1 = f1_score(y1_test, prediction)
        precision = precision_score(y1_test, prediction)
        avg_precision = average_precision_score(y1_test, prediction)

        # Salvataggio metriche
        metrics[model_name]["Accuracy"] = accuracy
        metrics[model_name]["F1-Score"] = f1
        metrics[model_name]["Precision"] = precision
        metrics[model_name]["Avg Precision"] = avg_precision

    # Creazione DataFrame per la visualizzazione
    metrics_df = pd.DataFrame(metrics).T

    # Plot delle metriche
    metrics_df.plot(kind="bar", figsize=(10, 6), colormap="viridis", edgecolor="black")
    plt.title("Confronto delle Metriche tra Modelli")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(loc="lower right")
    plt.show()
    plt.close()
    pass

# Funzione per Random Forest
def RF(X1, y1):
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, train_size=0.7, random_state=13)

    # Aggiunta di scaling per Random Forest
    scaler = StandardScaler()
    X1_train = scaler.fit_transform(X1_train)
    X1_test = scaler.transform(X1_test)

    clf = RandomForestClassifier(n_estimators=20, random_state=0)
    clf = clf.fit(X1_train, y1_train)

    prediction = clf.predict(X1_test)
    accuracy = accuracy_score(prediction, y1_test)

    # Train modello con cross-validation di 5 fold
    cv_scores = cross_val_score(clf, X1, y1, cv=5)

    print("\n--- Random Forest ---")
    print(f"Media dei punteggi: {np.mean(cv_scores):.3f}")
    print(f"Varianza dei punteggi: {np.var(cv_scores):.3f}")
    print(f"Deviazione standard dei punteggi: {np.std(cv_scores):.3f}")
    print("\n")

    # AUC
    probs = clf.predict_proba(X1_test)[:, 1]
    auc = roc_auc_score(y1_test, probs)
    print(f"AUC: {auc:.3f}")

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y1_test, probs)
    pyplot.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
    pyplot.plot(fpr, tpr, marker='.', label='Random Forest')
    pyplot.xlabel('FPR (False Positive Rate)')
    pyplot.ylabel('TPR (True Positive Rate)')
    pyplot.title('ROC Curve')
    pyplot.legend()
    pyplot.show()
    plt.close()
    pass


    # Precision-Recall Curve
    average_precision = average_precision_score(y1_test, prediction)
    precision, recall, _ = precision_recall_curve(y1_test, prediction)
    
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall Curve: AP={average_precision:.2f}')
    plt.show()
    plt.close()
    pass


    # Stampa il Classification Report e Confusion Matrix
    print("\n--- Classification Report ---")
    print(classification_report(y1_test, prediction))

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y1_test, prediction)
    print(cm)

    # Visualizza la Confusion Matrix come heatmap
    df_cm = pd.DataFrame(cm, index=['Classe 0', 'Classe 1'], columns=['Classe 0', 'Classe 1'])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'size': 16})
    plt.title('Confusion Matrix')
    plt.show()
    plt.close()
    pass


    # Calcolo F1-Score
    f1 = f1_score(y1_test, prediction)
    
    # Stampa i risultati finali
    print("\n--- Performance Summary ---")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Average Precision: {average_precision:.3f}")
    print(f"F1-Score: {f1:.3f}")


# Funzione per Decision Tree
def DT(X1, y1):
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, train_size=0.7, random_state=13)

    # Aggiunta di scaling per Decision Tree
    scaler = StandardScaler()
    X1_train = scaler.fit_transform(X1_train)
    X1_test = scaler.transform(X1_test)

    clf = DecisionTreeClassifier(random_state=0)
    clf = clf.fit(X1_train, y1_train)

    prediction = clf.predict(X1_test)
    accuracy = accuracy_score(prediction, y1_test)

    # Train modello con cross-validation di 5 fold
    cv_scores = cross_val_score(clf, X1, y1, cv=5)

    print("\n--- Decision Tree ---")
    print(f"Media dei punteggi: {np.mean(cv_scores):.3f}")
    print(f"Varianza dei punteggi: {np.var(cv_scores):.3f}")
    print(f"Deviazione standard dei punteggi: {np.std(cv_scores):.3f}")
    print("\n")

    # AUC
    probs = clf.predict_proba(X1_test)[:, 1]
    auc = roc_auc_score(y1_test, probs)
    print(f"AUC: {auc:.3f}")

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y1_test, probs)
    pyplot.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
    pyplot.plot(fpr, tpr, marker='.', label='Decision Tree')
    pyplot.xlabel('FPR (False Positive Rate)')
    pyplot.ylabel('TPR (True Positive Rate)')
    pyplot.title('ROC Curve')
    pyplot.legend()
    pyplot.show()
    plt.close()
    pass


    # Precision-Recall Curve
    average_precision = average_precision_score(y1_test, prediction)
    precision, recall, _ = precision_recall_curve(y1_test, prediction)
    
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall Curve: AP={average_precision:.2f}')
    plt.show()
    plt.close()
    pass


    # Stampa il Classification Report e Confusion Matrix
    print("\n--- Classification Report ---")
    print(classification_report(y1_test, prediction))

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y1_test, prediction)
    print(cm)

    # Visualizza la Confusion Matrix come heatmap
    df_cm = pd.DataFrame(cm, index=['Classe 0', 'Classe 1'], columns=['Classe 0', 'Classe 1'])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'size': 16})
    plt.title('Confusion Matrix')
    plt.show()
    plt.close()
    pass


    # Calcolo F1-Score
    f1 = f1_score(y1_test, prediction)
    
    # Stampa i risultati finali
    print("\n--- Performance Summary ---")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Average Precision: {average_precision:.3f}")
    print(f"F1-Score: {f1:.3f}")


# Funzione per Logistic Regression
def LR(X1, y1):
    # Standardizzare i dati
    scaler = StandardScaler()
    X1_scaled = scaler.fit_transform(X1)

    # Split dei dati
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1_scaled, y1, train_size=0.7, random_state=13)

    # Modifica il max_iter e il solver (prova anche 'liblinear' o 'saga' se non funziona con 'lbfgs')
    clf = LogisticRegression(random_state=0, max_iter=50000, solver='liblinear')

    
    # Adattamento del modello
    clf = clf.fit(X1_train, y1_train)

    prediction = clf.predict(X1_test)
    accuracy = accuracy_score(prediction, y1_test)

    # Train modello con cross-validation di 5 fold
    cv_scores = cross_val_score(clf, X1_scaled, y1, cv=5)

    print("\n--- Logistic Regression ---")
    print(f"Media dei punteggi: {np.mean(cv_scores):.3f}")
    print(f"Varianza dei punteggi: {np.var(cv_scores):.3f}")
    print(f"Deviazione standard dei punteggi: {np.std(cv_scores):.3f}")
    print("\n")

    # AUC
    probs = clf.predict_proba(X1_test)[:, 1]
    auc = roc_auc_score(y1_test, probs)
    print(f"AUC: {auc:.3f}")

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y1_test, probs)
    pyplot.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
    pyplot.plot(fpr, tpr, marker='.', label='Logistic Regression')
    pyplot.xlabel('FPR (False Positive Rate)')
    pyplot.ylabel('TPR (True Positive Rate)')
    pyplot.title('ROC Curve')
    pyplot.legend()
    pyplot.show()
    plt.close()
    pass


    # Precision-Recall Curve
    average_precision = average_precision_score(y1_test, prediction)
    precision, recall, _ = precision_recall_curve(y1_test, prediction)
    
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall Curve: AP={average_precision:.2f}')
    plt.show()
    plt.close()
    pass


    # Stampa il Classification Report e Confusion Matrix
    print("\n--- Classification Report ---")
    print(classification_report(y1_test, prediction))

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y1_test, prediction)
    print(cm)

    # Visualizza la Confusion Matrix come heatmap
    df_cm = pd.DataFrame(cm, index=['Classe 0', 'Classe 1'], columns=['Classe 0', 'Classe 1'])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'size': 16})
    plt.title('Confusion Matrix')
    plt.show()
    plt.close()
    pass


    # Calcolo F1-Score
    f1 = f1_score(y1_test, prediction)
    
    # Stampa i risultati finali
    print("\n--- Performance Summary ---")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Average Precision: {average_precision:.3f}")
    print(f"F1-Score: {f1:.3f}")

# Applicazione dei tre modelli
def plot_all_learning_curves(X1, y1):
    """
    Funzione per tracciare le curve di apprendimento di tutti i modelli su un unico grafico.
    """
    plt.figure(figsize=(10, 7))

    # Modelli
    rf_model = RandomForestClassifier(n_estimators=20, random_state=0)
    dt_model = DecisionTreeClassifier(random_state=0)
    lr_model = LogisticRegression(random_state=0, max_iter=50000, solver='liblinear')


    # Calcola le curve di apprendimento
    models = [("Random Forest", rf_model), ("Decision Tree", dt_model), ("Logistic Regression", lr_model)]
    
    for model_name, model in models:
        train_sizes, train_scores, test_scores = learning_curve(
            model, X1, y1, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
        )

        train_mean = train_scores.mean(axis=1)
        test_mean = test_scores.mean(axis=1)

        # Traccia le curve di apprendimento
        plt.plot(train_sizes, train_mean, label=f'{model_name} (Training score)', marker="o")
        plt.plot(train_sizes, test_mean, label=f'{model_name} (Validation score)', marker="o")

    # Personalizza il grafico
    plt.title("Learning Curves for All Models")
    plt.xlabel("Training Size")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
    plt.close()
    pass

