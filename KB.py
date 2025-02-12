import csv
from pyswip import Prolog
from itertools import islice
import tkinter as tk  # Importa Tkinter per la GUI
from tkinter import scrolledtext  # Per aggiungere un'area di testo scorrevole

class PrologKBGenerator:
    def __init__(self, input_file, output_file):
        self.input_file = input_file  # Il file CSV di input
        self.output_file = output_file  # Il file KB di output (.pl)
        self.data = []  # Per memorizzare i dati del dataset
        self.columns = []  # Per memorizzare i nomi delle colonne
        self.feature_mapping = {}  # Mappatura delle feature (Feature1 -> NomeColonna)

    def load_data(self):
        """
        Carica i dati dal file CSV e li memorizza nella lista self.data.
        Salva anche i nomi delle colonne nella lista self.columns.
        """
        with open(self.input_file, mode='r') as file:
            reader = csv.reader(file)
            self.columns = next(reader)  # Leggi i nomi delle colonne
            for row in reader:
                self.data.append(row)

        # Crea la mappatura delle feature (Feature1 -> NomeColonna)
        # Partiamo dalla terza colonna (indice 2) perché le prime due sono ID e Diagnosis
        self.feature_mapping = {
            f"Feature{i}": self.columns[i + 1] for i in range(1, len(self.columns) - 1)
        }

    def generate_kb(self):
        """
        Genera il Knowledge Base in formato Prolog e lo scrive nel file di output.
        Ogni record nel dataset sarà convertito in una tupla Prolog.
        """
        with open(self.output_file, 'w') as file:
            for row in self.data:
                # Convertiamo 'M' in 'm' e 'B' in 'b' per Prolog
                diagnosis = row[1].lower()  # 'M' diventa 'm' e 'B' diventa 'b'
                
                try:
                    # Scriviamo il fatto in formato Prolog
                    fact = f"data({row[0]}, {diagnosis}, {', '.join(map(str, row[2:]))})."
                    file.write(fact + "\n")  # Scrivi nel file
                except Exception as e:
                    print(f"Errore nel generare il fatto per la riga {row}: {e}")

    def query_kb(self, query, limit=5):
        """
        Esegui una query sul KB Prolog e limita i risultati a un massimo di `limit`.
        
        :param query: La query in formato stringa da eseguire sul KB
        :param limit: Numero massimo di risultati da restituire (default: 5)
        :return: I risultati della query (massimo `limit` risultati)
        """
        prolog = Prolog()
        
        # Carica il file KB Prolog
        prolog.consult(self.output_file)
        
        # Esegui la query e limita i risultati
        result = list(islice(prolog.query(query), limit))
        return result

# Funzione per estrarre il nome della colonna dalla query
def extract_column_from_query(query):
    """
    Estrae il nome della colonna dalla query.
    Ad esempio, se la query è "Feature1 > 10.0", restituisce "Feature1".
    """
    # Trova la parte della query che contiene la condizione
    condition = query.split(",")[-1].strip()
    # Estrae il nome della colonna
    column = condition.split()[0]
    return column

# Funzione per stampare i risultati in una finestra Tkinter
def print_results_to_window(results, columns, feature_mapping, query_description, text_widget):
    """
    Stampa i risultati in un widget Text di Tkinter.
    
    :param results: I risultati della query
    :param columns: Le colonne della query (es. ["Feature1", "Feature2"])
    :param feature_mapping: La mappatura delle feature
    :param query_description: Descrizione della query
    :param text_widget: Il widget Text di Tkinter in cui stampare i risultati
    """
    if not results:
        text_widget.insert(tk.END, f"{query_description}\nNessun risultato trovato.\n\n")
        return
    
    # Stampa l'intestazione della query
    text_widget.insert(tk.END, "=" * 60 + "\n")
    text_widget.insert(tk.END, query_description + "\n")
    text_widget.insert(tk.END, "=" * 60 + "\n")
    text_widget.insert(tk.END, f"Trovati {len(results)} risultati:\n\n")
    
    # Stampa i risultati
    for i, result in enumerate(results, start=1):
        text_widget.insert(tk.END, f"Risultato {i}:\n")
        text_widget.insert(tk.END, f"  ID: {result['ID']}\n")
        for column in columns:
            column_name = feature_mapping.get(column, column)
            text_widget.insert(tk.END, f"  {column_name}: {float(result.get(column, 0)):.4f}\n")
        text_widget.insert(tk.END, "-" * 40 + "\n")  # Separatore tra i risultati
    text_widget.insert(tk.END, "\n")

# Funzione principale per eseguire le query e visualizzare i risultati
def run_queries_and_display():
    input_file = "data.csv"  # Assicurati che il tuo file CSV sia nel formato corretto
    output_file = "kb.pl"   # Il file dove vuoi scrivere il KB Prolog
    
    # Creazione dell'istanza della classe
    kb_generator = PrologKBGenerator(input_file, output_file)
    
    # Caricamento dei dati dal file CSV
    kb_generator.load_data()
    
    # Generazione del KB in formato Prolog
    kb_generator.generate_kb()

    # Creazione della finestra Tkinter
    window = tk.Tk()
    window.title("Risultati delle Query")
    window.geometry("800x700")  # Dimensioni della finestra

    # Aggiungi un'area di testo scorrevole
    text_widget = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=100, height=30)
    text_widget.pack(padx=10, pady=10)

    # Definizione delle query disponibili
    queries = [
        {
            "description": "Query 1: Campioni con radius_mean > 15.0",
            "query": "data(ID, Diagnosis, Feature1, Feature2, Feature3, Feature4, Feature5, Feature6, Feature7, Feature8, Feature9, Feature10, Feature11, Feature12, Feature13, Feature14, Feature15, Feature16, Feature17, Feature18, Feature19, Feature20, Feature21, Feature22, Feature23, Feature24, Feature25, Feature26, Feature27, Feature28, Feature29, Feature30), Feature1 > 15.0.",
            "columns": ["Feature1"]
        },
        {
            "description": "Query 2: Campioni con texture_mean < 20.0",
            "query": "data(ID, Diagnosis, Feature1, Feature2, Feature3, Feature4, Feature5, Feature6, Feature7, Feature8, Feature9, Feature10, Feature11, Feature12, Feature13, Feature14, Feature15, Feature16, Feature17, Feature18, Feature19, Feature20, Feature21, Feature22, Feature23, Feature24, Feature25, Feature26, Feature27, Feature28, Feature29, Feature30), Feature2 < 20.0.",
            "columns": ["Feature2"]
        },
        {
            "description": "Query 3: Campioni con area_mean > 800.0 e compactness_mean > 0.1",
            "query": "data(ID, Diagnosis, Feature1, Feature2, Feature3, Feature4, Feature5, Feature6, Feature7, Feature8, Feature9, Feature10, Feature11, Feature12, Feature13, Feature14, Feature15, Feature16, Feature17, Feature18, Feature19, Feature20, Feature21, Feature22, Feature23, Feature24, Feature25, Feature26, Feature27, Feature28, Feature29, Feature30), Feature4 > 800.0, Feature6 > 0.1.",
            "columns": ["Feature4", "Feature6"]
        },
        {
            "description": "Query 4: Campioni con perimeter_mean > 100.0 e concavity_mean > 0.2",
            "query": "data(ID, Diagnosis, Feature1, Feature2, Feature3, Feature4, Feature5, Feature6, Feature7, Feature8, Feature9, Feature10, Feature11, Feature12, Feature13, Feature14, Feature15, Feature16, Feature17, Feature18, Feature19, Feature20, Feature21, Feature22, Feature23, Feature24, Feature25, Feature26, Feature27, Feature28, Feature29, Feature30), Feature3 > 100.0, Feature8 > 0.2.",
            "columns": ["Feature3", "Feature8"]
        }
    ]

    # Funzione per eseguire una query selezionata
    def execute_query(query_info):
        # Pulisci l'area di testo prima di eseguire una nuova query
        text_widget.delete(1.0, tk.END)
        
        # Esegui la query
        results = kb_generator.query_kb(query_info["query"], limit=5)
        
        # Stampa i risultati
        print_results_to_window(results, query_info["columns"], kb_generator.feature_mapping, query_info["description"], text_widget)

    # Aggiungi pulsanti per ogni query
    for query_info in queries:
        button = tk.Button(window, text=query_info["description"], command=lambda q=query_info: execute_query(q))
        button.pack(pady=5)

    # Avvia la finestra Tkinter
    window.mainloop()

# Esegui la funzione principale
if __name__ == "__main__":
    run_queries_and_display()