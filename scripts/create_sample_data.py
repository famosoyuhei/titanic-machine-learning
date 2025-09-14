"""
Create sample Titanic dataset for development
タイタニックデータセットのサンプルを作成
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_titanic_data():
    """Create sample Titanic training and test data"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create sample training data (891 rows like original)
    n_train = 891
    
    # Basic features
    passenger_ids = list(range(1, n_train + 1))
    
    # Pclass: 1, 2, 3 (ticket class)
    pclass = np.random.choice([1, 2, 3], n_train, p=[0.24, 0.21, 0.55])
    
    # Sex: male, female
    sex = np.random.choice(['male', 'female'], n_train, p=[0.65, 0.35])
    
    # Age: realistic age distribution
    age = np.random.normal(29.7, 14.5, n_train)
    age = np.clip(age, 0.42, 80)  # Clip to realistic range
    age[np.random.choice(n_train, int(0.2 * n_train), replace=False)] = np.nan  # 20% missing
    
    # SibSp: Number of siblings/spouses
    sibsp = np.random.choice([0, 1, 2, 3, 4, 5, 8], n_train, p=[0.68, 0.23, 0.06, 0.02, 0.005, 0.003, 0.002])
    
    # Parch: Number of parents/children
    parch = np.random.choice([0, 1, 2, 3, 4, 5, 6], n_train, p=[0.76, 0.13, 0.08, 0.02, 0.005, 0.004, 0.001])
    
    # Fare: ticket fare
    fare_by_class = {1: (84, 80), 2: (20, 15), 3: (13, 12)}
    fare = []
    for pc in pclass:
        mean_fare, std_fare = fare_by_class[pc]
        f = np.random.normal(mean_fare, std_fare)
        fare.append(max(0, f))
    fare = np.array(fare)
    
    # Embarked: C, Q, S
    embarked = np.random.choice(['C', 'Q', 'S'], n_train, p=[0.19, 0.09, 0.72])
    embarked[np.random.choice(n_train, 2, replace=False)] = np.nan  # 2 missing values
    
    # Names (simplified)
    titles = ['Mr.', 'Mrs.', 'Miss.', 'Master.', 'Dr.', 'Rev.']
    names = []
    for i, s in enumerate(sex):
        if s == 'male':
            title = np.random.choice(['Mr.', 'Master.', 'Dr.', 'Rev.'], p=[0.8, 0.15, 0.03, 0.02])
        else:
            title = np.random.choice(['Mrs.', 'Miss.'], p=[0.6, 0.4])
        names.append(f"Passenger_{i+1}, {title} Sample Name")
    
    # Tickets (simplified)
    tickets = [f"TICKET_{i+1}" for i in range(n_train)]
    
    # Cabin (mostly missing)
    cabins = [f"C{np.random.randint(1, 100)}" if np.random.random() < 0.23 else np.nan for _ in range(n_train)]
    
    # Survived: target variable (realistic survival rates)
    # Higher survival for females, higher class
    survival_prob = []
    for s, pc in zip(sex, pclass):
        base_prob = 0.38  # Overall survival rate
        if s == 'female':
            base_prob += 0.35
        if pc == 1:
            base_prob += 0.25
        elif pc == 2:
            base_prob += 0.1
        survival_prob.append(min(0.95, max(0.05, base_prob)))
    
    survived = np.random.binomial(1, survival_prob)
    
    # Create DataFrame
    train_data = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': survived,
        'Pclass': pclass,
        'Name': names,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Ticket': tickets,
        'Fare': fare,
        'Cabin': cabins,
        'Embarked': embarked
    })
    
    # Create test data (418 rows like original)
    n_test = 418
    
    # Similar process for test data (without Survived column)
    test_passenger_ids = list(range(n_train + 1, n_train + n_test + 1))
    test_pclass = np.random.choice([1, 2, 3], n_test, p=[0.24, 0.21, 0.55])
    test_sex = np.random.choice(['male', 'female'], n_test, p=[0.65, 0.35])
    
    test_age = np.random.normal(29.7, 14.5, n_test)
    test_age = np.clip(test_age, 0.42, 80)
    test_age[np.random.choice(n_test, int(0.2 * n_test), replace=False)] = np.nan
    
    test_sibsp = np.random.choice([0, 1, 2, 3, 4, 5, 8], n_test, p=[0.68, 0.23, 0.06, 0.02, 0.005, 0.003, 0.002])
    test_parch = np.random.choice([0, 1, 2, 3, 4, 5, 6], n_test, p=[0.76, 0.13, 0.08, 0.02, 0.005, 0.004, 0.001])
    
    test_fare = []
    for pc in test_pclass:
        mean_fare, std_fare = fare_by_class[pc]
        f = np.random.normal(mean_fare, std_fare)
        test_fare.append(max(0, f))
    test_fare = np.array(test_fare)
    
    test_embarked = np.random.choice(['C', 'Q', 'S'], n_test, p=[0.19, 0.09, 0.72])
    
    test_names = []
    for i, s in enumerate(test_sex):
        if s == 'male':
            title = np.random.choice(['Mr.', 'Master.', 'Dr.', 'Rev.'], p=[0.8, 0.15, 0.03, 0.02])
        else:
            title = np.random.choice(['Mrs.', 'Miss.'], p=[0.6, 0.4])
        test_names.append(f"TestPassenger_{i+1}, {title} Test Name")
    
    test_tickets = [f"TEST_TICKET_{i+1}" for i in range(n_test)]
    test_cabins = [f"C{np.random.randint(1, 100)}" if np.random.random() < 0.23 else np.nan for _ in range(n_test)]
    
    test_data = pd.DataFrame({
        'PassengerId': test_passenger_ids,
        'Pclass': test_pclass,
        'Name': test_names,
        'Sex': test_sex,
        'Age': test_age,
        'SibSp': test_sibsp,
        'Parch': test_parch,
        'Ticket': test_tickets,
        'Fare': test_fare,
        'Cabin': test_cabins,
        'Embarked': test_embarked
    })
    
    return train_data, test_data

def save_sample_data():
    """Save sample data to CSV files"""
    
    print("Creating sample Titanic dataset...")
    train_df, test_df = create_sample_titanic_data()
    
    # Create data directory path
    data_dir = Path("../data")
    data_dir.mkdir(exist_ok=True)
    
    # Save to CSV
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"[OK] Training data saved: {train_path}")
    print(f"   Shape: {train_df.shape}")
    print(f"   Survival rate: {train_df['Survived'].mean():.2%}")
    
    print(f"[OK] Test data saved: {test_path}")
    print(f"   Shape: {test_df.shape}")
    
    # Display basic info
    print(f"\nTraining data info:")
    print(train_df.info())
    
    print(f"\nTarget variable distribution:")
    print(train_df['Survived'].value_counts())
    
    return train_df, test_df

if __name__ == "__main__":
    save_sample_data()