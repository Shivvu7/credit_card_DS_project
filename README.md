![Alt text](https://repository-images.githubusercontent.com/223699949/0601d980-5912-11eb-8486-3237a1e3c4cf)

# 🛡️ Credit Card Fraud Detection

## 📌 Project Overview  
This project focuses on detecting fraudulent credit card transactions through **data preprocessing**, **analysis**, **visualization**, and **machine learning models**. By exploring and modeling transaction data, the project aims to identify patterns and improve fraud detection accuracy.

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
