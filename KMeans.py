# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score
from inspect import signature

# Funzione per applicare il metodo del gomito e evidenziare il k ottimale
def elbow_method(X):
    inertia = []
    silhouette_scores = []  
    k_range = range(2, 11)  

    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=5, n_init=10, random_state=0)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
        silhouette_avg = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    ax[0].plot(k_range, inertia, marker='o', color='b', linestyle='-', label='Inertia')
    ax[0].set_title('Metodo del Gomito')
    ax[0].set_xlabel('Numero di Cluster (k)')
    ax[0].set_ylabel('Inertia')
    ax[0].grid(True)
    optimal_k_inertia = np.argmin(inertia) + 2  
    ax[0].plot(optimal_k_inertia, inertia[optimal_k_inertia-2], 'ro', label=f'k={optimal_k_inertia} (Inertia)')
    ax[0].legend()

    ax[1].plot(k_range, silhouette_scores, marker='o', color='g', linestyle='-', label='Silhouette Score')
    ax[1].set_title('Silhouette Score')
    ax[1].set_xlabel('Numero di Cluster (k)')
    ax[1].set_ylabel('Silhouette Score')
    ax[1].grid(True)
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]  
    ax[1].plot(optimal_k_silhouette, silhouette_scores[np.argmax(silhouette_scores)], 'ro', label=f'k={optimal_k_silhouette} (Silhouette)')
    ax[1].legend()

    plt.show()
    plt.close()
    optimal_k = optimal_k_silhouette
    print(f"\nNumero ottimale di cluster (k): {optimal_k}")

    return optimal_k

# Funzione per visualizzare la distribuzione delle etichette di diagnosis
def KMEANS(X, y):
    class_counts = np.bincount(y)
    plt.figure(figsize=(8, 6))
    plt.pie(class_counts, labels=[f'Classe {i}' for i in range(len(class_counts))], autopct='%1.1f%%', startangle=90)
    plt.title('Distribuzione delle Etichette Reali (Prima del Clustering)')
    plt.axis('equal')  
    plt.show()
    plt.close()

    optimal_k = elbow_method(X)

    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    y_kmeans = kmeans.fit_predict(X)  

    cluster_counts = np.bincount(y_kmeans)
    plt.figure(figsize=(8, 6))
    plt.pie(cluster_counts, labels=[f'Cluster {i}' for i in range(optimal_k)], autopct='%1.1f%%', startangle=90)
    plt.title(f'Distribuzione dei Cluster con k={optimal_k}')
    plt.axis('equal')  
    plt.show()
    plt.close()

    average_precision = average_precision_score(y, y_kmeans)
    precision, recall, _ = precision_recall_curve(y, y_kmeans)

    step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall Curve: AP={average_precision:.2f}')
    plt.show()
    plt.close()

    accuracy = accuracy_score(y, y_kmeans)
    precision = precision_score(y, y_kmeans, average='weighted')
    recall = recall_score(y, y_kmeans, average='weighted')
    f1 = f1_score(y, y_kmeans, average='weighted')

    print("\n--- Performance Summary ---")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"Average Precision: {average_precision:.3f}")

    print("\n--- Classification Report ---")
    print(classification_report(y, y_kmeans))

    confusion_Matrix = confusion_matrix(y, y_kmeans)
    print("\n--- Confusion Matrix ---")
    print(confusion_Matrix)

    df_cm = pd.DataFrame(confusion_Matrix, index=['Classe 0', 'Classe 1'], columns=['Classe 0', 'Classe 1'])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'size': 16})
    plt.title('Confusion Matrix')
    plt.show()
    plt.close()

    # Grafico a barre per confrontare Accuracy, Precision, Recall, F1-score, Average Precision
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Avg Precision': average_precision
    }

    plt.figure(figsize=(10, 6))
    plt.bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.ylim(0, 1)  
    plt.xlabel('Metriche')
    plt.ylabel('Valore')
    plt.title('Confronto delle Metriche')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, v in enumerate(metrics.values()):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12)

    plt.show()
    plt.close()
