import tkinter as tk
from tkinter import ttk, messagebox
from constraint import Problem

specialisti = ["Dr. Rossi (Radiologo)", "Dr. Bianchi (Chirurgo)", "Dr. Verdi (Oncologo)"]
giorni_settimana = ["lunedi", "martedi", "mercoledi", "giovedi", "venerdi", "sabato"]
turni_disponibili = ["9:00-12:00", "12:00-15:00", "15:00-18:00"]

class MedicoCSP(Problem):
    def __init__(self, specialista):
        super().__init__()
        self.specialista = specialista
        self.addVariable("giorno", giorni_settimana)
        self.addVariable("orario", turni_disponibili)
        
    def get_disponibilita(self):
        soluzioni = self.getSolutions()
        giorni = sorted(set(s["giorno"] for s in soluzioni))
        orari = sorted(set(s["orario"] for s in soluzioni))
        return giorni, orari

def crea_problemi():
    medici = [MedicoCSP(specialisti[i]) for i in range(3)]
    medici[0].addConstraint(lambda g, o: o in ["9:00-12:00", "12:00-15:00"] if g == "lunedi" else o in ["15:00-18:00"], ["giorno", "orario"])
    medici[1].addConstraint(lambda g, o: o in ["9:00-12:00", "12:00-15:00"] if g == "martedi" else o in ["15:00-18:00"], ["giorno", "orario"])
    medici[2].addConstraint(lambda g, o: o in ["9:00-12:00", "12:00-15:00"] if g == "giovedi" else o in ["15:00-18:00"], ["giorno", "orario"])
    return medici

class PrenotazioneGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Prenotazione Esami")
        self.root.geometry("200x200")  # Aumentata la dimensione della finestra
        self.medici = crea_problemi()

        ttk.Label(root, text="Seleziona lo specialista:").pack()
        self.specialista_var = tk.StringVar()
        self.specialista_menu = ttk.Combobox(root, textvariable=self.specialista_var, values=specialisti)
        self.specialista_menu.pack()
        
        self.btn_disponibilita = ttk.Button(root, text="Mostra disponibilità", command=self.mostra_disponibilita)
        self.btn_disponibilita.pack()
        
        self.giorno_var = tk.StringVar()
        self.orario_var = tk.StringVar()
        self.giorno_menu = ttk.Combobox(root, textvariable=self.giorno_var)
        self.orario_menu = ttk.Combobox(root, textvariable=self.orario_var)
        
        self.btn_prenota = ttk.Button(root, text="Prenota", command=self.prenota, state=tk.DISABLED)
        self.btn_prenota.pack()
        
    def mostra_disponibilita(self):
        index = specialisti.index(self.specialista_var.get())
        giorni, orari = self.medici[index].get_disponibilita()
        
        if giorni and orari:
            self.giorno_menu.config(values=giorni)
            self.orario_menu.config(values=orari)
            self.giorno_menu.pack()
            self.orario_menu.pack()
            self.btn_prenota.config(state=tk.NORMAL)
        else:
            messagebox.showerror("Errore", "Nessuna disponibilità per questo specialista.")

    def prenota(self):
        messagebox.showinfo("Prenotazione Confermata", f"Prenotato con {self.specialista_var.get()}\nGiorno: {self.giorno_var.get()}\nOrario: {self.orario_var.get()}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PrenotazioneGUI(root)
    root.mainloop()
