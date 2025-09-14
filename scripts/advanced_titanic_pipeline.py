"""
Advanced Titanic Machine Learning Pipeline
高度なタイタニック機械学習パイプライン - スコア最大化版
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                             ExtraTreesClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif

class AdvancedTitanicPipeline:
    """Advanced Titanic survival prediction pipeline"""
    
    def __init__(self, data_dir="../data"):
        self.data_dir = Path(data_dir)
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.models = {}
        self.results = {}
        self.scaler = RobustScaler()
        
    def load_data(self):
        """Load and initial data exploration"""
        print("=== Loading Titanic Data ===")
        
        train_path = self.data_dir / "train.csv"
        test_path = self.data_dir / "test.csv"
        
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
        print(f"Training data: {self.train_df.shape}")
        print(f"Test data: {self.test_df.shape}")
        print(f"Survival rate: {self.train_df['Survived'].mean():.2%}")
        
        return self.train_df, self.test_df
    
    def advanced_feature_engineering(self):
        """Advanced feature engineering with domain expertise"""
        print("=== Advanced Feature Engineering ===")
        
        # Combine datasets for consistent preprocessing
        train_size = len(self.train_df)
        df_combined = pd.concat([
            self.train_df.drop(['Survived'], axis=1),
            self.test_df
        ], ignore_index=True, sort=False)
        
        # 1. Title extraction with detailed mapping
        df_combined['Title'] = df_combined['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        
        # Advanced title mapping
        title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
            'Dr': 'Professional', 'Rev': 'Professional', 'Col': 'Military', 
            'Major': 'Military', 'Mlle': 'Miss', 'Countess': 'Nobility', 
            'Ms': 'Miss', 'Lady': 'Nobility', 'Jonkheer': 'Nobility', 
            'Don': 'Nobility', 'Dona': 'Nobility', 'Mme': 'Mrs',
            'Capt': 'Military', 'Sir': 'Nobility'
        }
        df_combined['Title'] = df_combined['Title'].map(title_mapping).fillna('Rare')
        
        # 2. Advanced family features
        df_combined['FamilySize'] = df_combined['SibSp'] + df_combined['Parch'] + 1
        df_combined['IsAlone'] = (df_combined['FamilySize'] == 1).astype(int)
        df_combined['SmallFamily'] = ((df_combined['FamilySize'] >= 2) & (df_combined['FamilySize'] <= 4)).astype(int)
        df_combined['LargeFamily'] = (df_combined['FamilySize'] >= 5).astype(int)
        
        # 3. Advanced age processing
        # Age imputation based on multiple features
        age_groups = df_combined.groupby(['Title', 'Pclass', 'Sex'])['Age'].median()
        
        for idx, row in df_combined.iterrows():
            if pd.isna(row['Age']):
                title, pclass, sex = row['Title'], row['Pclass'], row['Sex']
                
                # Try multiple fallbacks
                if (title, pclass, sex) in age_groups:
                    df_combined.loc[idx, 'Age'] = age_groups[(title, pclass, sex)]
                elif (title, pclass) in df_combined.groupby(['Title', 'Pclass'])['Age'].median():
                    df_combined.loc[idx, 'Age'] = df_combined.groupby(['Title', 'Pclass'])['Age'].median()[(title, pclass)]
                else:
                    df_combined.loc[idx, 'Age'] = df_combined['Age'].median()
        
        # Age categories
        df_combined['Child'] = (df_combined['Age'] < 16).astype(int)
        df_combined['Adult'] = ((df_combined['Age'] >= 16) & (df_combined['Age'] < 60)).astype(int)
        df_combined['Senior'] = (df_combined['Age'] >= 60).astype(int)
        df_combined['AgeGroup'] = pd.cut(df_combined['Age'], bins=[0, 16, 32, 48, 64, 100], 
                                       labels=['Child', 'Young', 'Adult', 'Middle', 'Senior'])
        
        # 4. Advanced fare processing
        df_combined['Fare'].fillna(df_combined.groupby(['Pclass', 'Embarked'])['Fare'].median().median(), inplace=True)
        df_combined['FarePerPerson'] = df_combined['Fare'] / df_combined['FamilySize']
        df_combined['ExpensiveFare'] = (df_combined['Fare'] > df_combined['Fare'].quantile(0.75)).astype(int)
        df_combined['CheapFare'] = (df_combined['Fare'] < df_combined['Fare'].quantile(0.25)).astype(int)
        
        # 5. Advanced cabin processing
        df_combined['HasCabin'] = df_combined['Cabin'].notna().astype(int)
        df_combined['CabinDeck'] = df_combined['Cabin'].str[0].fillna('Unknown')
        
        # Deck survival rates (historical knowledge)
        deck_mapping = {
            'A': 'Upper', 'B': 'Upper', 'C': 'Upper', 
            'D': 'Middle', 'E': 'Middle', 'F': 'Lower', 
            'G': 'Lower', 'T': 'Special', 'Unknown': 'Unknown'
        }
        df_combined['CabinLevel'] = df_combined['CabinDeck'].map(deck_mapping)
        
        # 6. Advanced ticket processing
        df_combined['TicketPrefix'] = df_combined['Ticket'].str.extract('([A-Za-z]+)', expand=False).fillna('None')
        df_combined['TicketNumber'] = df_combined['Ticket'].str.extract('(\\d+)', expand=False).fillna('0').astype(int)
        df_combined['SharedTicket'] = df_combined.groupby('Ticket')['Ticket'].transform('count') > 1
        df_combined['SharedTicket'] = df_combined['SharedTicket'].astype(int)
        
        # 7. Social class indicators
        df_combined['HighClass'] = ((df_combined['Pclass'] == 1) | 
                                   (df_combined['Title'].isin(['Mrs', 'Miss', 'Master'])) |
                                   (df_combined['Fare'] > df_combined['Fare'].quantile(0.8))).astype(int)
        
        # 8. Survival likelihood features (based on historical patterns)
        df_combined['WomanOrChild'] = ((df_combined['Sex'] == 'female') | (df_combined['Age'] < 16)).astype(int)
        df_combined['MaleAdult'] = ((df_combined['Sex'] == 'male') & (df_combined['Age'] >= 16)).astype(int)
        
        # 9. Embarked processing
        df_combined['Embarked'].fillna(df_combined['Embarked'].mode()[0], inplace=True)
        
        print(f"Feature engineering completed!")
        print(f"Original features: {len(self.train_df.columns) - 1}")
        print(f"Engineered features: {len([col for col in df_combined.columns if col not in self.train_df.columns])}")
        
        # Select features for modeling
        feature_cols = [
            # Original numerical features
            'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
            # Engineered numerical features
            'FamilySize', 'IsAlone', 'SmallFamily', 'LargeFamily',
            'Child', 'Adult', 'Senior', 'FarePerPerson', 'ExpensiveFare', 'CheapFare',
            'HasCabin', 'SharedTicket', 'HighClass', 'WomanOrChild', 'MaleAdult',
            # Categorical features (will be encoded)
            'Sex', 'Embarked', 'Title', 'AgeGroup', 'CabinLevel', 'TicketPrefix'
        ]
        
        # Encode categorical variables
        categorical_cols = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'CabinLevel', 'TicketPrefix']
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_combined[col] = le.fit_transform(df_combined[col].astype(str))
        
        # Split back to train and test
        self.X_train = df_combined[:train_size][feature_cols]
        self.X_test = df_combined[train_size:][feature_cols]
        self.y_train = self.train_df['Survived']
        
        print(f"Final training features: {self.X_train.shape}")
        print(f"Final test features: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train
    
    def feature_selection(self, k_features=20):
        """Select top k features"""
        print(f"=== Feature Selection (top {k_features}) ===")
        
        selector = SelectKBest(score_func=f_classif, k=k_features)
        X_train_selected = selector.fit_transform(self.X_train, self.y_train)
        X_test_selected = selector.transform(self.X_test)
        
        selected_features = self.X_train.columns[selector.get_support()]
        print(f"Selected features: {list(selected_features)}")
        
        return X_train_selected, X_test_selected, selected_features
    
    def train_advanced_models(self):
        """Train ensemble of advanced models with hyperparameter tuning"""
        print("=== Training Advanced Model Ensemble ===")
        
        # Scale features for some models
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        
        # Define advanced models with initial parameters
        models = {
            'rf_optimized': RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_split=5,
                min_samples_leaf=2, random_state=42
            ),
            'gb_optimized': GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=200, max_depth=8, min_samples_split=5, random_state=42
            ),
            'logistic_optimized': LogisticRegression(
                C=1.0, penalty='l2', solver='liblinear', random_state=42
            ),
            'svm_optimized': SVC(
                C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42
            ),
            'mlp_optimized': MLPClassifier(
                hidden_layer_sizes=(100, 50), alpha=0.01, learning_rate='adaptive',
                random_state=42, max_iter=500
            )
        }
        
        # Cross-validation with stratified folds
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Train and evaluate models
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Choose scaled or original features
            X_train_used = X_train_scaled if name in ['svm_optimized', 'mlp_optimized', 'logistic_optimized'] else self.X_train
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_used, self.y_train, cv=cv, scoring='accuracy')
            
            # Fit model
            model.fit(X_train_used, self.y_train)
            
            # Train predictions
            train_pred = model.predict(X_train_used)
            train_accuracy = accuracy_score(self.y_train, train_pred)
            
            # Store results
            self.models[name] = model
            self.results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_accuracy': train_accuracy,
                'scaled_features': name in ['svm_optimized', 'mlp_optimized', 'logistic_optimized']
            }
            
            print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"  Train Accuracy: {train_accuracy:.4f}")
        
        # Rank models by CV performance
        sorted_models = dict(sorted(self.results.items(), key=lambda x: x[1]['cv_mean'], reverse=True))
        
        print(f"\\nModel Performance Ranking:")
        for i, (name, results) in enumerate(sorted_models.items(), 1):
            print(f"{i:2d}. {name:20s}: CV={results['cv_mean']:.4f} (+/-{results['cv_std']*2:.4f})")
        
        return self.models, self.results
    
    def create_voting_ensemble(self, top_k=5):
        """Create voting ensemble from top k models"""
        print(f"\\n=== Creating Voting Ensemble (top {top_k} models) ===")
        
        # Get top k models
        sorted_models = sorted(self.results.items(), key=lambda x: x[1]['cv_mean'], reverse=True)[:top_k]
        
        ensemble_models = []
        for name, results in sorted_models:
            ensemble_models.append((name, self.models[name]))
            print(f"Including: {name} (CV: {results['cv_mean']:.4f})")
        
        # Create voting classifier
        voting_clf = VotingClassifier(
            estimators=ensemble_models,
            voting='soft'  # Use probability predictions
        )
        
        # Train ensemble
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(voting_clf, self.X_train, self.y_train, cv=cv, scoring='accuracy')
        
        voting_clf.fit(self.X_train, self.y_train)
        
        print(f"Ensemble CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.models['voting_ensemble'] = voting_clf
        self.results['voting_ensemble'] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_accuracy': accuracy_score(self.y_train, voting_clf.predict(self.X_train)),
            'scaled_features': False
        }
        
        return voting_clf
    
    def make_final_predictions(self, model_name='voting_ensemble'):
        """Generate final predictions"""
        print(f"\\n=== Making Final Predictions ===")
        
        model = self.models[model_name]
        
        # Prepare test features
        if self.results[model_name]['scaled_features']:
            X_test_used = self.scaler.transform(self.X_test)
        else:
            X_test_used = self.X_test
        
        # Make predictions
        predictions = model.predict(X_test_used)
        prediction_proba = model.predict_proba(X_test_used)[:, 1]
        
        # Create submission
        submission = pd.DataFrame({
            'PassengerId': self.test_df['PassengerId'],
            'Survived': predictions
        })
        
        print(f"Model used: {model_name}")
        print(f"CV Performance: {self.results[model_name]['cv_mean']:.4f}")
        print(f"Test survival rate: {predictions.mean():.2%}")
        
        return submission, prediction_proba
    
    def save_results(self, submission, filename='advanced_titanic_submission.csv'):
        """Save submission and results"""
        output_path = self.data_dir.parent / 'results' / filename
        submission.to_csv(output_path, index=False)
        
        print(f"\\nSubmission saved: {output_path}")
        print(f"Ready for Kaggle submission!")
        
        return output_path

def main():
    """Main execution pipeline"""
    print("=" * 50)
    print("ADVANCED TITANIC SURVIVAL PREDICTION PIPELINE")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = AdvancedTitanicPipeline()
    
    # Load data
    pipeline.load_data()
    
    # Feature engineering
    pipeline.advanced_feature_engineering()
    
    # Train models
    pipeline.train_advanced_models()
    
    # Create ensemble
    pipeline.create_voting_ensemble()
    
    # Make predictions
    submission, probabilities = pipeline.make_final_predictions()
    
    # Save results
    pipeline.save_results(submission)
    
    # Performance summary
    best_single_model = max(pipeline.results.items(), key=lambda x: x[1]['cv_mean'] if x[0] != 'voting_ensemble' else 0)
    ensemble_performance = pipeline.results['voting_ensemble']['cv_mean']
    
    print(f"\\n" + "=" * 50)
    print(f"ADVANCED PIPELINE RESULTS SUMMARY")
    print(f"=" * 50)
    print(f"Best Single Model: {best_single_model[0]} ({best_single_model[1]['cv_mean']:.4f})")
    print(f"Voting Ensemble:   {ensemble_performance:.4f}")
    print(f"Performance Gain:  {ensemble_performance - best_single_model[1]['cv_mean']:+.4f}")
    print(f"Final Features:    {pipeline.X_train.shape[1]}")
    print(f"Test Predictions:  {len(submission)} passengers")
    
    return pipeline, submission

if __name__ == "__main__":
    pipeline, submission = main()