# 🧪 Wine Type Detection using Neural Networks 🍷
This project is a binary classification model that detects whether a given wine is red or white based on its chemical properties. It uses data from the UCI Wine Quality dataset, performs basic visualization, and trains a neural network using Keras.

# 📁 Files
Wine_type_detection.ipynb: Jupyter Notebook containing the full code, from data loading to model training and evaluation.

🧰 Technologies Used
1. Python
2. Pandas & NumPy – Data handling
3. Matplotlib – Visualization 📊
4. scikit-learn – Train-test split
5. TensorFlow/Keras – Model creation & training

# 📊 Data Description
The dataset contains features like:

1. Alcohol
2. Acidity
3. Sulphates
4. Chlorides
5. Density
...and more.

Two datasets are merged:

. Red wine labeled as 1
. White wine labeled as 0

# 📌 Key Steps
1. 📥 Data Loading
```python
red = pd.read_csv(...red wine data...)
white = pd.read_csv(...white wine data...)
```
2. 🧹 Data Preparation
Add a type column: 1 for red, 0 for white.

Combine datasets.

Drop missing values.

3. 📊 Visualization
Histograms of alcohol content by wine type to show distribution.

4. 🔄 Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=45)
```
5. 🧠 Neural Network Model
```python
model = Sequential()
model.add(Dense(12, activation='relu', input_dim=12))
model.add(Dense(9, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```
6. ⚙️ Compile & Train
```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3, batch_size=1, verbose=1)
```
✅ Output
The model is trained to classify red vs. white wines.

Accuracy is displayed per epoch.

Can be evaluated further with metrics like confusion matrix, precision, recall, etc.

🚀 How to Run
Clone the repository

Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
```
Run the Jupyter notebook:

```bash
jupyter notebook Wine_type_detection.ipynb
```
