![Alt text](https://repository-images.githubusercontent.com/223699949/0601d980-5912-11eb-8486-3237a1e3c4cf)



## 📌 Project Overview  
This project focuses on detecting fraudulent credit card transactions through **data preprocessing**, **analysis**, **visualization**, and **machine learning models**. By exploring and modeling transaction data, the project aims to identify patterns and improve fraud detection accuracy.  

### **Key Aspects Covered in This Project:**  
✅ **Data Collection & Preprocessing** – Handling imbalanced datasets, removing noise, and feature engineering.  
✅ **Exploratory Data Analysis (EDA)** – Identifying transaction trends, feature correlations, and fraud patterns using visualizations.  
✅ **Feature Selection & Engineering** – Transforming transaction data for better model performance.  
✅ **Machine Learning Models** – Implementing algorithms like Logistic Regression, Random Forest, XGBoost, and Neural Networks.  
✅ **Anomaly Detection Techniques** – Using Isolation Forests, Autoencoders, and Unsupervised Learning for detecting rare fraud cases.  
✅ **Model Evaluation & Optimization** – Hyperparameter tuning, cross-validation, and performance metrics (Precision, Recall, F1-score, AUC-ROC).  
✅ **Real-Time Fraud Detection** – Simulating live transaction monitoring using streaming data techniques.  
✅ **Deployment** – Integrating the model into a web or cloud-based API for real-world applications.  


![Alt text](https://files.oaiusercontent.com/file-L2SuEkxsLykbngva14zFwf?se=2025-02-06T14%3A43%3A05Z&sp=r&sv=2024-08-04&sr=b&rscc=max-age%3D604800%2C%20immutable%2C%20private&rscd=attachment%3B%20filename%3D0dc150c7-7337-4e84-818a-322f7d9f93f4.webp&sig=sISEzA6ztPC45oqq6ElX4mv6CrmDBP1i92EKC7vF8RI%3D)


## 📊 Dataset  
The dataset used in this project was **prepared and curated by me**. It includes transaction records with features like **amount, time**, and **anonymized attributes** that provide insights for fraud detection.  

📂 **Dataset Link**: [Click Here](https://github.com/Shivvu7/credit_dataset_sivasai/blob/main/dataset.csv.xlsx)

## 📂 Project Structure

1. **📥 Data Reading & Preprocessing**:
   - 🧹 Load the dataset and clean data (handling missing values & duplicates).
   - 📏 Normalize and prepare the data for analysis and model input.

2. **📊 Data Analysis**:
   - 🔍 Explore the dataset to understand **fraud vs. non-fraud** transactions.
   - 📈 Compute key statistics, such as the **percentage of fraudulent transactions**.

3. **📊 Data Visualization**:
   - 📌 Visualize the **frequency** of fraudulent and non-fraudulent transactions.
   - 📉 Analyze **transaction amount distributions** for both categories to identify fraud patterns.

4. **🤖 Model Development**:
   - ✂️ Split the dataset into **training and testing sets**.
   - 🏋️ Train and evaluate **machine learning models** to classify transactions.
   - 📊 Assess model performance and suggest **improvements for fraud detection**.

   



## ⚙️ Requirements  
Install the necessary dependencies using:
```bash
pip install -r requirements.txt

where requirements.txt includes: ✔️ pandas
✔️ numpy
✔️ matplotlib
✔️ seaborn
✔️ scikit-learn

🚀 Usage
📂 Load Data: Place the dataset in the project directory and load it using pandas.
🔍 Preprocess & Analyze: Run the scripts to clean data and extract fraud patterns.
📊 Visualize: Generate charts & plots to compare fraudulent and non-fraudulent transactions.
🤖 Train Model: Train and evaluate machine learning models for fraud detection.
📊 Results
✅ The dataset reveals key fraud detection insights, including fraud distribution and transaction patterns.
✅ The best-performing model accurately classifies fraudulent and legitimate transactions, providing a strong foundation for further enhancements.


## Results
* Key findings include the proportion of fraudulent transactions and insights on transaction amount distribution.
* The best-performing model accurately distinguishes between fraud and legitimate transactions, providing a foundation for further enhancements in fraud detection.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
