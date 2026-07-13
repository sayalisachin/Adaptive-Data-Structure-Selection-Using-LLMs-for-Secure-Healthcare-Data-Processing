# Hybrid Intelligent Healthcare Disease Prediction System

A hybrid intelligent healthcare diagnosis framework that integrates **Association Rule Mining**, **Graph-Based Inference**, and **Similarity-Based Prediction** to provide accurate and explainable disease predictions from patient symptoms.

---

## 📖 Overview

Healthcare diagnosis often involves incomplete symptom information, making disease prediction challenging. Traditional machine learning models generally rely on complete datasets and often fail to explain their predictions.

This project proposes a hybrid framework that combines data mining and graph analytics to improve disease prediction. The system discovers symptom associations using the Apriori algorithm, represents them as a weighted graph, infers missing symptoms through graph traversal, and predicts diseases using cosine similarity.

---

## ✨ Features

- Association Rule Mining using Apriori Algorithm
- Frequent Itemset Generation
- Rule Evaluation using:
  - Support
  - Confidence
  - Lift
  - Conviction
  - Cosine Similarity
- Weighted Adjacency List Graph Construction
- Graph-Based Symptom Inference using Breadth-First Search (BFS)
- Similarity-Based Disease Prediction
- Explainable AI Predictions
- Handles Incomplete Patient Symptoms
- Confidence Score Generation

---

## 🏗 System Architecture

```
Healthcare Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Apriori Algorithm
        │
        ▼
Association Rule Generation
        │
        ▼
Weighted Graph Construction
        │
        ▼
Graph-Based Symptom Expansion (BFS)
        │
        ▼
Cosine Similarity Computation
        │
        ▼
Disease Ranking
        │
        ▼
Final Disease Prediction
```

---

## 📂 Dataset

**Dataset Name**

Healthcare Symptoms–Disease Classification Dataset

**Source**

Kaggle

**Dataset Type**

Synthetic Healthcare Dataset

**Attributes**

- Patient_ID
- Age
- Gender
- Symptoms
- Symptom_Count
- Disease

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Mlxtend
- NetworkX
- Matplotlib
- Jupyter Notebook

---

## 📦 Project Structure

```
Hybrid-Healthcare-Diagnosis/
│
├── dataset/
│   └── healthcare_dataset.csv
│
├── notebooks/
│   └── DiseasePrediction.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── apriori.py
│   ├── graph_builder.py
│   ├── inference.py
│   ├── similarity.py
│   └── prediction.py
│
├── outputs/
│   ├── association_rules.csv
│   ├── graph.png
│   └── predictions.csv
│
├── README.md
└── requirements.txt
```

---

## ⚙ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Hybrid-Healthcare-Diagnosis.git
```

Navigate into the project directory

```bash
cd Hybrid-Healthcare-Diagnosis
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the notebook or Python scripts.

---

## 📊 Workflow

1. Load Healthcare Dataset
2. Data Cleaning and Preprocessing
3. Symptom Encoding
4. Frequent Itemset Mining
5. Association Rule Generation
6. Graph Construction
7. Symptom Expansion using BFS
8. Cosine Similarity Calculation
9. Disease Ranking
10. Final Prediction

---

## 📈 Evaluation Metrics

The system evaluates prediction performance using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Association rules are evaluated using:

- Support
- Confidence
- Lift
- Conviction
- Cosine Similarity

---

## 🎯 Advantages

- Explainable disease prediction
- Handles incomplete symptom inputs
- Efficient graph traversal
- Reduced search space
- Improved diagnostic accuracy
- Scalable architecture
- Interpretable rule-based reasoning

---

## 🚀 Future Enhancements

- Integration with Large Language Models (LLMs)
- Adaptive Data Structure Selection
- Real-time Clinical Decision Support
- Electronic Health Record (EHR) Integration
- Streamlit-based Web Application
- Knowledge Graph Expansion
- Multi-Disease Prediction
- Deep Learning-based Disease Classification

---

## 📚 References

1. Agrawal, R., & Srikant, R. (1994). Fast Algorithms for Mining Association Rules.
2. Han, J., Kamber, M., & Pei, J. Data Mining: Concepts and Techniques.
3. Scikit-learn Documentation
4. Mlxtend Documentation
5. NetworkX Documentation

---

## 👩‍💻 Author

**Sayali Sachin Chorge**

As developed for Manipal Institute Of Technology, Research Paper, 2026.

---

## 📜 License

This project is developed for academic and research purposes.
