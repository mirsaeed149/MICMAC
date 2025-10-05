# MICMAC
Interactive MICMAC Analysis Tool built with Python and Streamlit. Upload multiple Excel files, calculate influence–dependence matrices, visualize variable dynamics, and identify driving, linkage, dependent, and autonomous factors — all in one simple web app.
[README.md](https://github.com/user-attachments/files/22706210/README.md)
# 🧩 MICMAC Structural Analysis App

An open-source **MICMAC (Matrix of Cross-Impact Multiplications Applied to Classification)** tool built in **Python + Streamlit**, designed to help researchers, strategists, and analysts explore system dynamics and identify key drivers and dependencies.

---

## 🚀 Features

✅ Upload multiple Excel files (from multiple experts or sources)  
✅ Automatically compute the **overall influence–dependence matrix**  
✅ Visualize results in an interactive **influence vs. dependence map**  
✅ Highlight **driving**, **linkage**, **dependent**, and **autonomous** variables  
✅ Export or download results as Excel reports (optional)  
✅ 100% local and private — runs in your browser via Streamlit  

---

## 📂 Folder Structure

```
micmac_app/
│
├── micmac_app.py              # Main Streamlit application
├── requirements.txt           # Python dependencies
├── sample_data/               # Example Excel files
│   ├── Expert_1.xlsx
│   ├── Expert_2.xlsx
│   └── ...
└── README.md
```

---

## 🧠 What is MICMAC?

MICMAC (Matrix of Cross-Impact Multiplications Applied to Classification) is a **structural analysis method** used to study how variables in a system influence one another.  
It helps classify factors into four groups:

| Type | Influence | Dependence | Meaning |
|------|------------|-------------|----------|
| **Driving Variables** | High | Low | Core levers that shape the system |
| **Linkage Variables** | High | High | Sensitive, unstable factors — manage carefully |
| **Dependent Variables** | Low | High | Consequences or outcomes |
| **Autonomous Variables** | Low | Low | Isolated, low-impact factors |

---

## 🧩 Example Use Cases

- Strategic foresight and scenario planning  
- Organizational capability mapping  
- Policy and governance system analysis  
- Startup ecosystem or innovation network analysis  

---

## ⚙️ Installation & Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/micmac_app.git
cd micmac_app
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app
```bash
streamlit run micmac_app.py
```

The app will open automatically in your browser (usually at http://localhost:8501).

---

## 🧮 Input Format

Each Excel file should contain a **square matrix** with:
- Columns and rows representing the same set of variables  
- Values between **0–3 or 0–5**, showing how much each variable influences another  
- The first row used as column headers, and the first column as variable names

You can upload **multiple Excel files** — the app will automatically compute the **overall average matrix**.

---

## 📊 Output

- A scatter plot showing **influence vs. dependence** of each variable  
- Classification table listing each variable type  
- (Optional) Downloadable overall matrix Excel file

---

## 🌐 Online Deployment (Optional)

You can deploy this app for free using **[Streamlit Community Cloud](https://share.streamlit.io)**:

1. Push this repository to your GitHub account  
2. Go to Streamlit Cloud and connect your repo  
3. Deploy the app (it will automatically install dependencies)  

You’ll get a public link like:  
`https://yourusername-micmac-app.streamlit.app`

---

## 🧑‍💻 Author

**Mir Saeed Musavi**  
Business strategist & systems thinker  
📍 Digital Health & Innovation Projects  
💬 Connect on https://www.linkedin.com/in/mir-saeed-musavi-129025b2/

---

## 📜 License

This project is licensed under the **MIT License** — free for personal and academic use.  
Feel free to fork, modify, and contribute.

---

## ⭐ Acknowledgments

Thanks to the open-source community and all experts contributing to structural analysis and systems thinking.  
Inspired by **Michel Godet’s** original MICMAC methodology.
And aLso thanks to Chat GPT
