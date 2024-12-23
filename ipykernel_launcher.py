{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "af8487e0-6ce4-4c6f-b157-51a452cb6be5",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-12-23 12:58:32.111 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\soura\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.metrics import classification_report, accuracy_score\n",
    "\n",
    "# Title of the Streamlit app\n",
    "st.title(\"Credit Card Fraud Detection\")\n",
    "\n",
    "# File upload\n",
    "uploaded_file = st.file_uploader(\"Upload your CSV file\", type=\"csv\")\n",
    "\n",
    "if uploaded_file:\n",
    "    # Load the dataset\n",
    "    df = pd.read_csv(uploaded_file)\n",
    "    \n",
    "    # Display dataset\n",
    "    st.subheader(\"Dataset Preview\")\n",
    "    st.write(df.head())\n",
    "\n",
    "    # Summary statistics\n",
    "    st.subheader(\"Dataset Summary\")\n",
    "    st.write(df.describe())\n",
    "\n",
    "    # Check for missing values\n",
    "    st.subheader(\"Missing Values\")\n",
    "    st.write(df.isnull().sum())\n",
    "\n",
    "    # Visualizations\n",
    "    st.subheader(\"Class Distribution\")\n",
    "    fig, ax = plt.subplots()\n",
    "    sns.countplot(x='Class', data=df, ax=ax)\n",
    "    st.pyplot(fig)\n",
    "\n",
    "    # Splitting data\n",
    "    st.subheader(\"Model Training\")\n",
    "    X = df.drop(columns=['Class', 'id'])\n",
    "    y = df['Class']\n",
    "    \n",
    "    # Train-test split\n",
    "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n",
    "\n",
    "    # Standardizing features\n",
    "    scaler = StandardScaler()\n",
    "    X_train = scaler.fit_transform(X_train)\n",
    "    X_test = scaler.transform(X_test)\n",
    "\n",
    "    # Model fitting\n",
    "    model = RandomForestClassifier(random_state=42)\n",
    "    model.fit(X_train, y_train)\n",
    "\n",
    "    # Predictions\n",
    "    y_pred = model.predict(X_test)\n",
    "\n",
    "    # Metrics\n",
    "    st.subheader(\"Model Performance\")\n",
    "    st.write(\"Accuracy:\", accuracy_score(y_test, y_pred))\n",
    "    st.text(\"Classification Report:\")\n",
    "    st.text(classification_report(y_test, y_pred))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f743bdd0-40ed-4f46-8429-98d820bdef2f",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
