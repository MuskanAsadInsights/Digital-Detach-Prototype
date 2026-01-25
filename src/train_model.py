import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

def categorize_addiction(score):
    if score <= 4:
        return 0
    elif score <= 7:
        return 1
    else:
        return 2

def run_training_pipeline():
    # 1. DATA LOADING
    data_path = 'data/teen_phone_addiction_dataset.csv'
    df = pd.read_csv(data_path)

    # 2. INITIAL CLEANING
    df = df.drop(columns=['ID', 'Name', 'Location'])
    df = df.dropna()

    # 3. TARGET CATEGORIZATION
    df['Addiction_Label'] = df['Addiction_Level'].apply(categorize_addiction)

    # 4. CATEGORICAL ENCODING
    le_gender = LabelEncoder()
    le_purpose = LabelEncoder()
    le_grade = LabelEncoder()

    df['Gender'] = le_gender.fit_transform(df['Gender'])
    df['Phone_Usage_Purpose'] = le_purpose.fit_transform(df['Phone_Usage_Purpose'])
    df['School_Grade'] = le_grade.fit_transform(df['School_Grade'])

    # 5. FEATURE SELECTION
    features = [
        'Age', 'Gender', 'School_Grade', 'Daily_Usage_Hours', 'Sleep_Hours',
        'Academic_Performance', 'Social_Interactions', 'Exercise_Hours',
        'Anxiety_Level', 'Depression_Level', 'Self_Esteem', 'Parental_Control',
        'Screen_Time_Before_Bed', 'Phone_Checks_Per_Day', 'Apps_Used_Daily',
        'Time_on_Social_Media', 'Time_on_Gaming', 'Time_on_Education',
        'Phone_Usage_Purpose', 'Family_Communication', 'Weekend_Usage_Hours'
    ]

    X = df[features]
    y = df['Addiction_Label']

    # 6. FEATURE SCALING
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 7. MODEL TRAINING
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=500, random_state=42)
    model.fit(X_train, y_train)

    # 8. EVALUATION
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Model Training Accuracy: {accuracy * 100:.2f}%")

    # 9. ASSET EXPORT
    os.makedirs('models', exist_ok=True)
    with open('models/addiction_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/gender_encoder.pkl', 'wb') as f:
        pickle.dump(le_gender, f)
    with open('models/purpose_encoder.pkl', 'wb') as f:
        pickle.dump(le_purpose, f)
    with open('models/grade_encoder.pkl', 'wb') as f:
        pickle.dump(le_grade, f)

    print("Model and preprocessing assets saved to the models directory.")

if __name__ == "__main__":
    run_training_pipeline()