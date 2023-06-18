import pickle
import numpy as np

# Define the mapping between class labels and their names
class_names = {
    0: 'Advocate',
    1: 'Arts',
    2: 'Automation Testing',
    3: 'Blockchain',
    4: 'Business Analyst',
    5: 'Civil Engineer',
    6: 'Data Science',
    7: 'Database',
    8: 'DevOps Engineer',
    9: 'DotNet Developer',
    10: 'ETL Developer',
    11: 'Electrical Engineering',
    12: 'HR',
    13: 'Hadoop',
    14: 'Health and fitness',
    15: 'Java Developer',
    16: 'Mechanical Engineer',
    17: 'Network Security Engineer',
    18: 'Operations Manager',
    19: 'PMO',
    20: 'Python Developer',
    21: 'SAP Developer',
    22: 'Sales',
    23: 'Testing',
    24: 'Web Designing'
}

# def predict_category(file):
def predict_category(text):
    # Load the saved model from disk
    # text = get_text(file)
    clf = "model.pkl"
    with open(clf, 'rb') as file:
        model = pickle.load(file)

    # Prepare the input text for prediction
    input_vector = np.array([text])

    # Transform the input text using the saved vectorizer
    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
        input_vector = vectorizer.transform(input_vector)

    # Make a prediction using the loaded model
    if hasattr(model, 'predict_proba'):
        # For classifiers with predict_proba method (e.g. logistic regression, Naive Bayes)
        prediction_prob = model.predict_proba(input_vector)[0]
        predicted_index = np.argmax(prediction_prob)
        predicted_label = class_names[model.classes_[predicted_index]]
        return predicted_label
    else:
        # For classifiers without predict_proba method (e.g. SVM)
        prediction = model.predict(input_vector)[0]
        predicted_label = class_names[prediction]
        return predicted_label