import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.utils import resample

class PassengerSurvivalModel:
    def __init__(self):
        """
        Initializes the Passenger Survival Model by loading data, preparing it,
        training a Random Forest classifier, and evaluating model performance.
        """
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_path, "Passanger.csv")
        self.data = pd.read_csv(csv_path)
        self._prepare_data()
        self._train_model()
        self._evaluate_model()
    
    def _prepare_data(self):
        """
        Prepares the dataset by handling missing values, encoding categorical variables,
        scaling numerical features, and balancing the dataset via upsampling.
        """
        # Selecting relevant features
        self.data = self.data[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Cabin"]]
        
        # Handling missing values
        self.data["Age"].fillna(self.data["Age"].median(), inplace=True)
        self.data["Fare"].fillna(self.data["Fare"].median(), inplace=True)
        self.data["Embarked"].fillna(self.data["Embarked"].mode()[0], inplace=True)
        self.data["Cabin"].fillna('Unknown', inplace=True)
        
        # Encoding categorical variables
        self.data.replace({'Sex': {'male': 0, 'female': 1}}, inplace=True)
        self.data.replace({'Embarked': {'C': 0, 'Q': 1, 'S': 2}}, inplace=True)
        self.data['Cabin'] = self.data['Cabin'].apply(lambda x: 0 if x == 'Unknown' else 1)
        
        # Scaling numerical features
        predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Cabin"]
        self.scaler = StandardScaler()
        self.data[predictors] = self.scaler.fit_transform(self.data[predictors])
        
        # Balancing dataset using upsampling
        majority_class = self.data[self.data["Survived"] == 0]
        minority_class = self.data[self.data["Survived"] == 1]
        minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)
        self.data = pd.concat([majority_class, minority_upsampled])
        
        # Splitting into training and test sets
        self.train, self.test = train_test_split(self.data, test_size=0.2, random_state=42)
    
    def _train_model(self):
        """
        Trains the Random Forest classifier using the prepared training dataset.
        """
        target = "Survived"
        self.model.fit(self.train.drop(columns=['Survived']), self.train[target])
    
    def predict_survival(self, pclass, sex, age, sibsp, parch, fare, embarked, cabin):
        """
        Predicts survival based on the given passenger features.
        """
        input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked, cabin]])
        input_data_scaled = self.scaler.transform(input_data)
        prediction = self.model.predict(input_data_scaled)[0]
        return prediction
    
    def _evaluate_model(self):
        """
        Evaluates model performance using accuracy, confusion matrix, and classification report.
        """
        predictions = self.model.predict(self.test.drop(columns=["Survived"]))
        print("Accuracy:", accuracy_score(self.test["Survived"], predictions))
        print("Confusion Matrix:\n", confusion_matrix(self.test["Survived"], predictions))
        print("Classification Report:\n", classification_report(self.test["Survived"], predictions))
    
    def feature_importance(self):
        """
        Plots feature importance based on the trained Random Forest model.
        """
        plt.figure(figsize=(10, 6))
        feature_importances = self.model.feature_importances_
        feature_names = self.train.drop(columns=["Survived"]).columns
        feat_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
        feat_importance_df = feat_importance_df.sort_values(by='Importance', ascending=False)
        sns.barplot(x=feat_importance_df['Importance'], y=feat_importance_df['Feature'], palette='viridis')
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        plt.title("Feature Importance in Random Forest Model")
        plt.show()
    
    def confusion_matrix_function(self):
        """
        Plots a confusion matrix heatmap for model evaluation.
        """
        cm = confusion_matrix(self.test["Survived"], self.model.predict(self.test.drop(columns=["Survived"])))
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Did Not Survive", "Survived"], yticklabels=["Did Not Survive", "Survived"])
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.title("Confusion Matrix")
        plt.show()
    
    def roc_curve(self):
        """
        Plots the Receiver Operating Characteristic (ROC) curve.
        """
        probabilities = self.model.predict_proba(self.test.drop(columns=["Survived"]))[:, 1]
        fpr, tpr, _ = roc_curve(self.test["Survived"], probabilities)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='b', label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.show()
    
    def precision_recall_curve(self):
        """
        Plots the Precision-Recall curve for model performance analysis.
        """
        probabilities = self.model.predict_proba(self.test.drop(columns=["Survived"]))[:, 1]
        precision, recall, _ = precision_recall_curve(self.test["Survived"], probabilities)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, marker='.', color='b')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.show()
