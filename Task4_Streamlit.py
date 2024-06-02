import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
from kneed import KneeLocator
from io import BytesIO

# File paths
train_file_path = r"C:\Users\suraj\OneDrive\Desktop\ML\train.xlsx"
test_file_path = r"C:\Users\suraj\OneDrive\Desktop\ML\test.xlsx"
raw_file_path = r"C:\Users\suraj\OneDrive\Desktop\ML\rawdata.xlsx"

st.set_option('deprecation.showPyplotGlobalUse', False)


# Task 1: Clustering
@st.cache_data
def load_training_data(file_path):
    return pd.read_excel(file_path)

@st.cache_data
def train_kmeans(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=9, random_state=42)
    kmeans.fit(scaled_features)
    return kmeans, scaler, scaled_features

def clustering_task():
    st.header("Task 1: Machine Learning - Clustering")
    
    # Load the training data
    df = load_training_data(train_file_path)
    
    # Prepare features, dropping the target column if it exists
    if 'target' in df.columns:
        features = df.drop(columns=['target'])
    else:
        features = df

    # Train KMeans model and standardize features
    kmeans, scaler, scaled_features = train_kmeans(features)
    df['cluster'] = kmeans.labels_

    # PCA for visualization
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)
    pca_df = pd.DataFrame(data=principal_components, columns=['principal_component_1', 'principal_component_2'])
    pca_df['cluster'] = kmeans.labels_

    st.write("Clusters visualization using PCA:")
    fig, ax = plt.subplots(figsize=(10, 7))
    for cluster in pca_df['cluster'].unique():
        clustered_data = pca_df[pca_df['cluster'] == cluster]
        ax.scatter(clustered_data['principal_component_1'], clustered_data['principal_component_2'], label=f'Cluster {cluster}')
    ax.set_title('Clusters visualization using PCA')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.legend()
    st.pyplot(fig)

    # Section for user input to identify cluster
    st.write("### Predict Cluster for a New Data Point")
    
    # Create input fields for each feature
    input_data = []
    for column in features.columns:
        input_val = st.number_input(f"Input for {column}", value=0.0)
        input_data.append(input_val)

    if st.button("Predict Cluster"):
        # Standardize the input data using the same scaler
        input_data_scaled = scaler.transform([input_data])
        
        # Predict the cluster
        predicted_cluster = kmeans.predict(input_data_scaled)[0]
        
        st.write(f"The given data point belongs to cluster: {predicted_cluster}")

# Task 2: Classification
@st.cache_data
def load_data(file_path):
    return pd.read_excel(file_path)

@st.cache_data
def preprocess_data(train_df, test_df):
    X = train_df.drop(columns=['target'])
    y = train_df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    test_X = scaler.transform(test_df)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=0)
    return X_train, X_test, y_train, y_test, test_X

@st.cache_data
def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=100),
        'Decision Trees': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(n_estimators=50),  # Reduce number of estimators
        'Naive Bayes': GaussianNB()
    }

    accuracy, precision, recall = {}, {}, {}

    for key in models.keys():
        models[key].fit(X_train, y_train)
        predictions = models[key].predict(X_test)
        accuracy[key] = accuracy_score(y_test, predictions)
        precision[key] = precision_score(y_test, predictions, average='macro')
        recall[key] = recall_score(y_test, predictions, average='macro')

    return models, accuracy, precision, recall

def classification_task():
    st.header("Task 2: Machine Learning - Classification")
    
    try:
        train_df = load_data(train_file_path)
        test_df = load_data(test_file_path)
    except PermissionError as e:
        st.error(f"PermissionError: {e}")
        return

    X_train, X_test, y_train, y_test, test_X = preprocess_data(train_df, test_df)

    models, accuracy, precision, recall = train_and_evaluate_models(X_train, y_train, X_test, y_test)

    df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
    df_model['Accuracy'] = accuracy.values()
    df_model['Precision'] = precision.values()
    df_model['Recall'] = recall.values()

    st.write("Model Performance:")
    st.write(df_model)

    ax = df_model.plot.barh()
    ax.legend(ncol=len(models.keys()), bbox_to_anchor=(0, 1), loc='lower left', prop={'size': 14})
    plt.tight_layout()
    st.pyplot()

    test_predictions = {key: model.predict(test_X) for key, model in models.items()}
    test_predictions_df = pd.DataFrame(test_predictions)
    st.write("Test Predictions:")
    st.write(test_predictions_df)

    # Convert DataFrame to Excel in memory
    excel_data = BytesIO()
    with pd.ExcelWriter(excel_data, engine='xlsxwriter') as writer:
        test_predictions_df.to_excel(writer, index=False)
    excel_data.seek(0)

    st.download_button("Download Predictions", data=excel_data, file_name="test_predictions.xlsx")

# Task 3: Python
def python_task():
    st.header("Task 3: Python")

    raw_data = pd.read_excel(raw_file_path)

    # Ensure date and time are strings
    raw_data['date'] = raw_data['date'].astype(str)
    raw_data['time'] = raw_data['time'].astype(str)

    # Convert date and time to datetime format
    raw_data['datetime'] = pd.to_datetime(raw_data['date'] + ' ' + raw_data['time'])

    # Normalize position values
    raw_data['position'] = raw_data['position'].str.lower()

    # Sort by datetime
    raw_data = raw_data.sort_values(by='datetime')

    # Derive total duration for each inside and outside position
    raw_data['next_datetime'] = raw_data['datetime'].shift(-1)
    raw_data['duration'] = (raw_data['next_datetime'] - raw_data['datetime']).dt.total_seconds()

    # Filter out the rows where next_datetime is NaT (last row)
    raw_data = raw_data[raw_data['next_datetime'].notna()]

    # Group by date and position to calculate total duration
    duration_df = raw_data.groupby([raw_data['datetime'].dt.date, 'position'])['duration'].sum().reset_index()
    duration_df.columns = ['date', 'position', 'total_duration']

    # Pivot the duration dataframe to get inside and outside durations in separate columns
    duration_pivot_df = duration_df.pivot(index='date', columns='position', values='total_duration').reset_index()
    duration_pivot_df = duration_pivot_df.rename_axis(None, axis=1).fillna(0)
    duration_pivot_df.columns = ['date', 'inside_duration', 'outside_duration']

    # Group by date and activity to count number of activities
    activity_count_df = raw_data.groupby([raw_data['datetime'].dt.date, 'activity']).size().reset_index(name='count')
    activity_count_df.columns = ['date', 'activity', 'count']

    # Pivot the activity count dataframe to get pick and place activities in separate columns
    activity_pivot_df = activity_count_df.pivot(index='date', columns='activity', values='count').reset_index()
    activity_pivot_df = activity_pivot_df.rename_axis(None, axis=1).fillna(0)
    activity_pivot_df.columns = ['date', 'pick_activities', 'place_activities']

    # Merge the duration and activity dataframes on the date column
    final_df = pd.merge(duration_pivot_df, activity_pivot_df, on='date', how='left').fillna(0)

    st.write("Processed Data:")
    st.write(final_df)

    # Convert DataFrame to Excel in memory
    excel_data = BytesIO()
    with pd.ExcelWriter(excel_data, engine='xlsxwriter') as writer:
        final_df.to_excel(writer, index=False)
    excel_data.seek(0)

    st.download_button("Download Processed Data", data=excel_data, file_name="output.xlsx")

# Streamlit app
def main():
    st.title('Tasks Dashboard')

    task_choice = st.sidebar.selectbox("Select Task", ["Clustering", "Classification", "Python"])

    if task_choice == "Clustering":
        clustering_task()
    elif task_choice == "Classification":
        classification_task()
    elif task_choice == "Python":
        python_task()

if __name__ == '__main__':
    main()
