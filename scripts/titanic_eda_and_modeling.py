"""
Titanic Competition - EDA and Machine Learning Pipeline
タイタニック競技 - 探索的データ分析と機械学習パイプライン
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings('ignore')

class TitanicAnalyzer:
    """Titanic data analysis and modeling pipeline"""
    
    def __init__(self, data_dir="../data"):
        self.data_dir = Path(data_dir)
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load training and test data"""
        print("Loading Titanic data...")
        
        train_path = self.data_dir / "train.csv"
        test_path = self.data_dir / "test.csv"
        
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
        print(f"Training data: {self.train_df.shape}")
        print(f"Test data: {self.test_df.shape}")
        
        return self.train_df, self.test_df
    
    def exploratory_analysis(self):
        """Perform exploratory data analysis"""
        print("\n=== Exploratory Data Analysis ===")
        
        # Basic information
        print("\nTraining data info:")
        print(self.train_df.info())
        
        print("\nMissing values:")
        missing = self.train_df.isnull().sum()
        print(missing[missing > 0])
        
        # Survival statistics
        print(f"\nOverall survival rate: {self.train_df['Survived'].mean():.2%}")
        
        # Survival by key features
        print("\nSurvival by Sex:")
        print(self.train_df.groupby('Sex')['Survived'].agg(['count', 'mean']))
        
        print("\nSurvival by Pclass:")
        print(self.train_df.groupby('Pclass')['Survived'].agg(['count', 'mean']))
        
        print("\nSurvival by Embarked:")
        print(self.train_df.groupby('Embarked')['Survived'].agg(['count', 'mean']))
        
        # Age analysis
        print(f"\nAge statistics:")
        print(self.train_df['Age'].describe())
        
        return self.train_df.describe()
    
    def feature_engineering(self):
        """Extract and engineer features"""
        print("\n=== Feature Engineering ===")
        
        # Combine train and test for consistent preprocessing
        # Keep track of train size
        train_size = len(self.train_df)
        
        # Create a copy and combine
        df_combined = pd.concat([
            self.train_df.drop(['Survived'], axis=1),
            self.test_df
        ], ignore_index=True, sort=False)
        
        print(f"Combined dataset shape: {df_combined.shape}")
        
        # 1. Extract title from names
        df_combined['Title'] = df_combined['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        
        # Group rare titles
        title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
            'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
            'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
            'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
            'Capt': 'Rare', 'Sir': 'Rare'
        }
        df_combined['Title'] = df_combined['Title'].map(title_mapping).fillna('Rare')
        
        # 2. Family size features
        df_combined['FamilySize'] = df_combined['SibSp'] + df_combined['Parch'] + 1
        df_combined['IsAlone'] = (df_combined['FamilySize'] == 1).astype(int)
        
        # 3. Age binning and imputation
        # Fill missing ages based on Title and Pclass
        age_mapping = df_combined.groupby(['Title', 'Pclass'])['Age'].median()
        
        for idx, row in df_combined.iterrows():
            if pd.isna(row['Age']):
                title = row['Title']
                pclass = row['Pclass']
                if (title, pclass) in age_mapping:
                    df_combined.loc[idx, 'Age'] = age_mapping[(title, pclass)]
                else:
                    df_combined.loc[idx, 'Age'] = df_combined['Age'].median()
        
        # Create age bands
        df_combined['AgeGroup'] = pd.cut(df_combined['Age'], bins=5, labels=['Child', 'Young', 'Adult', 'Middle', 'Senior'])
        
        # 4. Fare processing
        df_combined['Fare'].fillna(df_combined['Fare'].median(), inplace=True)
        df_combined['FareGroup'] = pd.qcut(df_combined['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])
        
        # 5. Embarked processing
        df_combined['Embarked'].fillna(df_combined['Embarked'].mode()[0], inplace=True)
        
        # 6. Cabin processing
        df_combined['HasCabin'] = df_combined['Cabin'].notna().astype(int)
        df_combined['CabinDeck'] = df_combined['Cabin'].str[0].fillna('Unknown')
        
        # Simplify cabin deck
        deck_mapping = {'A': 'ABC', 'B': 'ABC', 'C': 'ABC', 'D': 'DE', 'E': 'DE', 
                       'F': 'FG', 'G': 'FG', 'T': 'T', 'Unknown': 'Unknown'}
        df_combined['CabinDeck'] = df_combined['CabinDeck'].map(deck_mapping)
        
        # 7. Ticket features
        df_combined['TicketPrefix'] = df_combined['Ticket'].str.extract('([A-Za-z]+)', expand=False).fillna('None')
        df_combined['TicketPrefix'] = df_combined['TicketPrefix'].apply(
            lambda x: x if df_combined['TicketPrefix'].value_counts()[x] > 10 else 'Rare'
        )
        
        print("Feature engineering completed!")
        print(f"New features: Title, FamilySize, IsAlone, AgeGroup, FareGroup, HasCabin, CabinDeck, TicketPrefix")
        
        # Select features for modeling
        feature_cols = [
            'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
            'Title', 'FamilySize', 'IsAlone', 'AgeGroup', 'FareGroup', 
            'HasCabin', 'CabinDeck', 'TicketPrefix'
        ]
        
        # Encode categorical variables
        categorical_cols = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup', 'CabinDeck', 'TicketPrefix']
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_combined[col] = le.fit_transform(df_combined[col].astype(str))
        
        # Split back to train and test
        self.X_train = df_combined[:train_size][feature_cols]
        self.X_test = df_combined[train_size:][feature_cols]
        self.y_train = self.train_df['Survived']
        
        print(f"Training features shape: {self.X_train.shape}")
        print(f"Test features shape: {self.X_test.shape}")
        print(f"Feature columns: {feature_cols}")
        
        return self.X_train, self.X_test, self.y_train
    
    def train_models(self):
        """Train multiple machine learning models"""
        print("\n=== Training Models ===")
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
            
            # Fit model
            model.fit(self.X_train, self.y_train)
            
            # Train predictions
            train_pred = model.predict(self.X_train)
            train_accuracy = accuracy_score(self.y_train, train_pred)
            
            # Store results
            self.models[name] = model
            self.results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_accuracy': train_accuracy,
                'model': model
            }
            
            print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"  Train Accuracy: {train_accuracy:.4f}")
        
        # Find best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['cv_mean'])
        best_model = self.results[best_model_name]['model']
        
        print(f"\nBest model: {best_model_name}")
        print(f"Best CV score: {self.results[best_model_name]['cv_mean']:.4f}")
        
        return self.models, best_model_name
    
    def optimize_best_model(self, model_name='Random Forest'):
        """Optimize hyperparameters for the best model"""
        print(f"\n=== Optimizing {model_name} ===")
        
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42)
            
        elif model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [3, 5, 7]
            }
            model = GradientBoostingClassifier(random_state=42)
            
        else:
            print(f"No optimization defined for {model_name}")
            return self.models[model_name]
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Update best model
        self.models[f'Optimized_{model_name}'] = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def make_predictions(self, model_name=None):
        """Make predictions on test set"""
        print("\n=== Making Predictions ===")
        
        if model_name is None:
            # Use best model
            model_name = max(self.results.keys(), key=lambda x: self.results[x]['cv_mean'])
        
        if f'Optimized_{model_name}' in self.models:
            model = self.models[f'Optimized_{model_name}']
            used_model = f'Optimized_{model_name}'
        else:
            model = self.models[model_name]
            used_model = model_name
        
        print(f"Using model: {used_model}")
        
        # Make predictions
        predictions = model.predict(self.X_test)
        prediction_proba = model.predict_proba(self.X_test)[:, 1]  # Probability of survival
        
        # Create submission DataFrame
        submission = pd.DataFrame({
            'PassengerId': self.test_df['PassengerId'],
            'Survived': predictions
        })
        
        print(f"Predictions completed!")
        print(f"Survival rate in test set: {predictions.mean():.2%}")
        print(f"Prediction shape: {submission.shape}")
        
        return submission, prediction_proba
    
    def feature_importance(self, model_name='Random Forest'):
        """Display feature importance"""
        print(f"\n=== Feature Importance ({model_name}) ===")
        
        if f'Optimized_{model_name}' in self.models:
            model = self.models[f'Optimized_{model_name}']
        else:
            model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(importances)
            return importances
        else:
            print(f"Model {model_name} does not have feature importance")
            return None

def main():
    """Main execution function"""
    print("=== Titanic Competition Pipeline ===")
    
    # Initialize analyzer
    analyzer = TitanicAnalyzer()
    
    # Load data
    analyzer.load_data()
    
    # Exploratory analysis
    analyzer.exploratory_analysis()
    
    # Feature engineering
    analyzer.feature_engineering()
    
    # Train models
    models, best_model_name = analyzer.train_models()
    
    # Optimize best model
    optimized_model = analyzer.optimize_best_model(best_model_name)
    
    # Feature importance
    analyzer.feature_importance(best_model_name)
    
    # Make predictions
    submission, probabilities = analyzer.make_predictions(best_model_name)
    
    # Save submission file
    results_dir = Path("../results")
    results_dir.mkdir(exist_ok=True)
    
    submission_path = results_dir / "titanic_submission.csv"
    submission.to_csv(submission_path, index=False)
    
    print(f"\n[SUCCESS] Submission file saved: {submission_path}")
    print(f"Ready for Kaggle submission!")
    
    return analyzer, submission

if __name__ == "__main__":
    analyzer, submission = main()