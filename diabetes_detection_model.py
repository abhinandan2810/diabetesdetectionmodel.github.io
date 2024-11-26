import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Load Dataset
def load_data():
    url = "https://storage.googleapis.com/kagglesdsdata/datasets/228/482/diabetes.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241126%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241126T162310Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=047cceeb466cd31ff94e81248a28efde148e64082e52cb2e405b9f72113ef9b746e82fca5b401fb419149f3f44fc7a9a0f39807bd34dcb5be32a1fab59c7b72c013d2a2aa637c8ad7489eb9608225babebb191df5734007295aa90a2adf2b269d977f1ea52056e64846b841767e344bd27ff0f4a4f4029681b93523682a3c5758819cf90efb4e431fae7fbc7534c823f3ed2591262fd8cb5749f49e79e120a0c2c2bba49b490e69822f497aae7baef79b16b223f14084756f6ae3f5f7ba09d78ce125a88d16e06b76a9af812b6f08eae6fc1589c668165a0fb6658e7e17c92dd91d2565872c1f3833cefa8570ebdf3cc09376a2070ebe67bd1e1afe86b0e0af0"
    
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                    'DiabetesPedigreeFunction', 'Age', 'Outcome']
    data = pd.read_csv(url, header=0, names=column_names)
    return data

# Train Model
def train_model():
    global model, scaler
    data = load_data()

    # Features and target
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# Predict Function
def predict_diabetes():
    try:
        # Get input values
        user_data = [
            float(entry_pregnancies.get()),
            float(entry_glucose.get()),
            float(entry_blood_pressure.get()),
            float(entry_skin_thickness.get()),
            float(entry_insulin.get()),
            float(entry_bmi.get()),
            float(entry_dpf.get()),
            float(entry_age.get())
        ]

        # Scale input data and make prediction
        user_data = scaler.transform([user_data])
        prediction = model.predict(user_data)[0]

        # Show result
        result = "Diabetes Detected: Yes" if prediction == 1 else "Diabetes Detected: No"
        messagebox.showinfo("Prediction Result", result)
    except Exception as e:
        messagebox.showerror("Input Error", f"Invalid input: {e}")

# Show Data Insights (Visualization)
def show_insights():
    data = load_data()

    # Create Figure
    fig = Figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(111)
    data['Age'].plot(kind='hist', bins=20, color='skyblue', ax=ax)
    ax.set_title("Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")

    # Embed in tkinter
    canvas = FigureCanvasTkAgg(fig, master=insights_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# GUI Setup
def setup_gui():
    global entry_pregnancies, entry_glucose, entry_blood_pressure, entry_skin_thickness
    global entry_insulin, entry_bmi, entry_dpf, entry_age, insights_frame

    # Root Window
    root = tk.Tk()
    root.title("Diabetes Prediction System")
    root.geometry("800x600")

    # Tabs
    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True)

    # Prediction Tab
    prediction_frame = ttk.Frame(notebook)
    notebook.add(prediction_frame, text="Prediction")

    # Insights Tab
    insights_frame = ttk.Frame(notebook)
    notebook.add(insights_frame, text="Insights")

    # Prediction Tab Layout
    fields = [
        ("Pregnancies", "entry_pregnancies"),
        ("Glucose Level", "entry_glucose"),
        ("Blood Pressure", "entry_blood_pressure"),
        ("Skin Thickness", "entry_skin_thickness"),
        ("Insulin Level", "entry_insulin"),
        ("BMI", "entry_bmi"),
        ("Diabetes Pedigree Function", "entry_dpf"),
        ("Age", "entry_age"),
    ]
    for i, (label_text, var_name) in enumerate(fields):
        label = ttk.Label(prediction_frame, text=label_text)
        label.grid(row=i, column=0, padx=10, pady=5)
        entry = ttk.Entry(prediction_frame)
        entry.grid(row=i, column=1, padx=10, pady=5)
        globals()[var_name] = entry

    # Predict Button
    predict_button = ttk.Button(prediction_frame, text="Predict", command=predict_diabetes)
    predict_button.grid(row=len(fields), columnspan=2, pady=20)

    # Insights Tab Layout
    show_insights()

    # Run the GUI
    root.mainloop()

# Run the Program
if __name__ == "__main__":
    print("Training Model...")
    accuracy = train_model()
    print(f"Model trained with an accuracy of {accuracy * 100:.2f}%")
    setup_gui()
