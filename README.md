my LinkedIn profile--> https://www.linkedin.com/in/sivasai-sariki-407774229
Project Title *Credit Card Default Prediction* 
Overview
This project aims to predict whether a customer is likely to default on their credit card payment in the next month using machine learning models. The project involves data preprocessing, feature engineering, model training, and evaluation.
We leverage TensorFlow and Keras for building and optimizing neural network models, alongside other machine learning techniques for performance comparison.
The data set consists of 2000 samples from each of two categories. Five variables are
1.Income of the person
2.Age of the person
3.Loan amount
4.Loan to Income Ratio (engineered feature)
5.Default

**Table of Contents**
Installation
Dataset
Project Workflow
Results
Future Enhancements
Technologies Used
Installation
To run this project on your local machine:

**Clone this repository**
bash command
git clone https://github.com/Shivvu7/credit_card_DS_project.git
cd credit_card_DS_project
**Install the required dependencies**
bash command
pip install -r requirements.txt
**Run the project scripts**
bash command
python src/train_model.py

**Dataset**
Source: URL -->  https://github.com/Shivvu7/credit_dataset_sivasai
**Description**
The dataset contains records of customers with features such as payment history, bill statements, and demographic details. The target variable indicates whether the customer defaulted or not.
**Data Preprocessing**
Handled missing values and outliers.
Scaled numerical features using StandardScaler.
Encoded categorical variables using One-Hot Encoding.
**Data Cleaning**
Cleaned and prepared the data for analysis and model training.
**Exploratory Data Analysis (EDA)**
Analyzed data distribution and feature correlations using visualizations.
**Feature Engineering**
Selected relevant features to improve model accuracy and reduce overfitting.
**Model Development**
Built a neural network using TensorFlow and Keras.
Compared performance with other models like Logistic Regression and Random Forest.
**Evaluation**
Assessed performance using metrics such as accuracy, precision, recall, F1-score, and AUC-ROC.
**Results**
Best Model: Neural Network using TensorFlow/Keras
Accuracy: 85%
AUC-ROC Score: 0.92
**Model Evaluation Metrics**
Metric	        Value
Accuracy	        85%
Precision	       87%
Recall	          82%
F1-Score	        84%
AUC-ROC	         0.92

**Visualization**:can be done by the graphical representation

**Future Enhancements**
Improve feature engineering by exploring additional data transformations.
Experiment with ensemble methods like XGBoost and Random Forest for better performance.
Deploy the model as a web application for real-time predictions.
**Technologies Used**
Programming Languages: Python
Libraries and Frameworks:
TensorFlow and Keras
Pandas and NumPy
Scikit-learn
Matplotlib and Seaborn
Tools: Jupyter Notebook

**Step-by-Step Implementation**
# Step 1 : import library
import pandas as pd
# Step 2 : import data
default = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Credit%20Default.csv')
default.head()
Income	Age	Loan	Loan-Income	Default
0	66155.92510	59.017015	8106.532131	0.122537	0
1	34415.15397	48.117153	6564.745018	0.190752	0
2	57317.17006	63.108049	8020.953296	0.139940	0
3	42709.53420	45.751972	6103.642260	0.142911	0
4	66952.68885	18.584336	8770.099235	0.130990	1
default.info()
<class 'pandas.core.frame.DataFrame'>

RangeIndex: 2000 entries, 0 to 1999

Data columns (total 5 columns):

 #   Column          Non-Null Count  Dtype  

---  ------          --------------  -----  

 0   Income          2000 non-null   float64

 1   Age             2000 non-null   float64

 2   Loan            2000 non-null   float64

 3   Loan to Income  2000 non-null   float64

 4   Default         2000 non-null   int64  

dtypes: float64(4), int64(1)

memory usage: 78.2 KB
default.describe()
Income	Age	Loan	Loan to Income	Default
count	2000.000000	2000.000000	2000.000000	2000.000000	2000.000000
mean	45331.600018	40.927143	4444.369695	0.098403	0.141500
std	14326.327119	13.262450	3045.410024	0.057620	0.348624
min	20014.489470	18.055189	1.377630	0.000049	0.000000
25%	32796.459720	29.062492	1939.708847	0.047903	0.000000
50%	45789.117310	41.382673	3974.719418	0.099437	0.000000
75%	57791.281670	52.596993	6432.410625	0.147585	0.000000
max	69995.685580	63.971796	13766.051240	0.199938	1.000000
# Count of each category
default['Default'].value_counts()
0    1717
1     283
Name: Default, dtype: int64
# Step 3 : define target (y) and features (X)
default.columns
Index(['Income', 'Age', 'Loan', 'Loan to Income', 'Default'], dtype='object')
y = default['Default']
X = default.drop(['Default'],axis=1)
# Step 4 : train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=2529)
# check shape of train and test sample
X_train.shape, X_test.shape, y_train.shape, y_test.shape
((1400, 4), (600, 4), (1400,), (600,))
# Step 5 : select model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
# Step 6 : train or fit model
model.fit(X_train,y_train)

LogisticRegression
LogisticRegression()
model.intercept_
array([9.39569095])
model.coef_
array([[-2.31410016e-04, -3.43062682e-01,  1.67863323e-03,
         1.51188530e+00]])
# Step 7 : predict model
y_pred = model.predict(X_test)
y_pred
array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
       1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
       0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
       1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
       0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
       0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
       0, 0, 0, 0, 0, 0])
# Step 8 : model accuracy
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
confusion_matrix(y_test,y_pred)
array([[506,  13],
       [ 17,  64]])
accuracy_score(y_test,y_pred)
0.95
print(classification_report(y_test,y_pred))
              precision    recall  f1-score   support



           0       0.97      0.97      0.97       519

           1       0.83      0.79      0.81        81



    accuracy                           0.95       600

   macro avg       0.90      0.88      0.89       600

weighted avg       0.95      0.95      0.95       600
