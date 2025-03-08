import tkinter as tk
from tkinter import ttk, messagebox
from Passanger_model import PassengerSurvivalModel

class PassengerSurvivalGUI:
    def __init__(self):
        self.model = PassengerSurvivalModel()
        self.gui = tk.Tk()
        self.gui.title("Passenger Survival Predictor")
        self.gui.geometry("1280x720")
        self.gui.config(bg="oldlace")
        self.gui.resizable(False, False)
        
        self.setup_ui()
        self.gui.mainloop()
    
    def setup_ui(self):
        """Setup UI elements"""
        self.create_title()
        self.create_model_selector()
        self.create_main_form()
        self.create_submit_button()
        self.create_graph_button()
    
    def create_title(self):
        """Create application title"""
        title_label = tk.Label(self.gui, text="Passenger Survival Predictor", background="oldlace", foreground="black", font=("Helvetica", 25))
        title_label.place(x=450, y=50)
    
    def create_model_selector(self):
        """Create airplane model selection dropdown"""
        frame = tk.Frame(self.gui, cursor="hand2", bg="oldlace")
        frame.grid(column=5, row=8, padx=10, pady=20)
        
        select_model_label = tk.Label(frame, text="Select Airplane Model:", bg="oldlace", foreground="black", font=("Helvetica", 15))
        select_model_label.pack()
        
        self.n = tk.StringVar()
        self.select_model = ttk.Combobox(frame, width=20, textvariable=self.n, font=("Helvetica"))
        self.select_model['values'] = ('Boeing 737', 'Airbus A320', 'Convair CV-240')
        self.select_model.pack(anchor="center")
    
    def create_main_form(self):
        """Create the main input form"""
        main_frame = tk.Frame(self.gui, bg="lavender", relief="sunken", height=500, width=1200, borderwidth=2.5)
        main_frame.place(x=40, y=150)
        
        # Name
        tk.Label(self.gui, text="Name", background="lavender", font=("Helvetica", 15)).place(x=500, y=250)
        self.name_entry = tk.Entry(self.gui, width=15, font=("Helvetica"))
        self.name_entry.place(x=580, y=250)
        
        # Age
        tk.Label(self.gui, text="Age", background="lavender", font=("Helvetica", 15)).place(x=500, y=300)
        self.age_entry = tk.Spinbox(self.gui, from_=1, to=99, width=10, font=("Helvetica", 15))
        self.age_entry.place(x=580, y=300)
        
        # Class
        tk.Label(self.gui, text="Class", background="lavender", font=("Helvetica", 15)).place(x=500, y=350)
        self.class_var = tk.StringVar()
        self.class_entry = ttk.Combobox(self.gui, width=15, textvariable=self.class_var, font=("Helvetica", 15))
        self.class_entry['values'] = ('Economy', 'First', 'Business')
        self.class_entry.place(x=580, y=350)
        
        # Sex
        tk.Label(self.gui, text="Sex", background="lavender", font=("Helvetica", 15)).place(x=500, y=400)
        self.sex_var = tk.StringVar()
        self.sex_entry = ttk.Combobox(self.gui, width=15, textvariable=self.sex_var, font=("Helvetica", 15))
        self.sex_entry['values'] = ('Male', 'Female')
        self.sex_entry.place(x=580, y=400)
        
        # Number of Travellers
        tk.Label(self.gui, text="No of Travellers", background="lavender", font=("Helvetica", 15)).place(x=400, y=450)
        self.people_entry = tk.Spinbox(self.gui, from_=1, to=10, width=10, font=("Helvetica", 15))
        self.people_entry.place(x=580, y=450)
    
    def create_submit_button(self):
        """Create submit button with progress bar"""
        self.progress = ttk.Progressbar(self.gui, length=200, mode='determinate')
        self.progress.place(x=500, y=600)
        
        submit_button = tk.Button(self.gui, text="Submit", bg="beige", font=("Helvetica", 15), command=self.bar, relief="raised")
        submit_button.place(x=550, y=515)
    
    def bar(self):
        """Simulate progress bar before prediction"""
        import time
        for i in range(0, 101, 20):
            self.progress['value'] = i
            self.gui.update_idletasks()
            time.sleep(0.3)
        self.progress['value'] = 0
        self.send_values()
    
    def send_values(self):
        """Send input values for prediction"""
        name_value = self.name_entry.get()
        if not name_value:
            messagebox.showinfo("Invalid Input", "Name cannot be empty")
            return
        
        age_value = int(self.age_entry.get())
        class_value = {'Economy': 3, 'First': 2, 'Business': 1}.get(self.class_entry.get(), None)
        if class_value is None:
            messagebox.showinfo("Invalid Input", "Select one of three classes")
            return
        
        sex_value = {'Male': 0, 'Female': 1}.get(self.sex_entry.get(), None)
        if sex_value is None:
            messagebox.showinfo("Invalid Input", "Select One of the two Genders")
            return
        
        people_value_1 = int(self.people_entry.get())
        if not self.select_model.get():
            messagebox.showinfo("Invalid Input", "Please Select Model")
            return
        
        prediction = self.model.predict_survival(class_value, sex_value, age_value, people_value_1, 0, 150, 0, 1)
        msg = f"{name_value} might {'survive' if prediction == 1 else 'not survive'}"
        messagebox.showinfo("Survival Prediction", msg)
    
    def create_graph_button(self):
        """Create button to open graph window"""
        graph_button = tk.Button(self.gui, text="Diagrams", bg="lavender", font=("Helvetica", 15), command=self.graph_window, relief="raised")
        graph_button.place(x=1140, y=42)
    
    def graph_window(self):
        """Create a new window for visualizations"""
        graph_window = tk.Toplevel()
        graph_window.title("Graphical Analysis")
        
        buttons = [
            ("Feature Importance", self.model.feature_importance),
            ("Heat Map", self.model.confusion_matrix_function),
            ("ROC Curve", self.model.roc_curve),
            ("Precision Recall Curve", self.model.Precision_recall_curve)
        ]
        
        for text, command in buttons:
            tk.Button(graph_window, text=text, font=("Helvetica", 15), command=command).pack(side="left", expand=True, fill="both")

if __name__ == "__main__":
    PassengerSurvivalGUI()
