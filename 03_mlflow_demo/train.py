# train.py

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Chargement du dataset
X, y = load_iris(return_X_y=True)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Crée le dossier pour les artéfacts (graphes)
os.makedirs("plots", exist_ok=True)

# Début du tracking MLflow
with mlflow.start_run(run_name="logreg_iris"): #Bonne pratique : chaque expérience doit être encapsulée

    # 🔁 Reproductibilité et initialisation Bonne pratique : fixer les aléas pour des résultats stables
    model = LogisticRegression(max_iter=200, random_state=42)

    # Entraînement
    model.fit(X_train, y_train)

    # Prédiction
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # 🧾 Logging : param, mtrics, artéfacts et modèles
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 200)
    mlflow.log_metric("accuracy", acc)

    # Signature pour déploiement
    signature = infer_signature(X_test, y_pred)

    # Log du modèle
    mlflow.sklearn.log_model(
        model, "model", signature=signature, input_example=X_test[:1]
    )

    # 🎨 Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    # Sauvegarde + log
    fig_path = "plots/confusion_matrix.png"
    plt.savefig(fig_path)
    mlflow.log_artifact(fig_path)

print("✅ Entraînement terminé. Visualise sur http://localhost:5000")
