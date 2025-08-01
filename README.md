# Review-Sentiment-Regression-Neural-Network
Predicting review sentiment as positive (true or false) using logistic regression and neural network
# Overview
The project's goal is to predict the sentiment of book reviews, classifying them as either positive or negative. The project follows the complete machine learning lifecycle, from data selection and exploratory data analysis to model implementation, evaluation, and optimization.
The two models explored in this project are Logistic Regression & Neural Network
# Objectives & Goals
The main objective of this project is to build and evaluate a machine learning model that can accurately predict the sentiment of a given book review.

The specific goals are:
- Select and prepare a suitable dataset: Choose one of the provided datasets and perform necessary data cleaning and preprocessing.
- Define a clear ML problem: Formulate a supervised learning problem for sentiment classification.
- Implement and compare two models: Train and evaluate both a Logistic Regression model and a Neural Network.
- Optimize model performance: Tune the hyperparameters of the models to improve their accuracy and generalization.
- Document the process: Provide a clear and detailed explanation of the steps taken, from data preparation to final results.
# Installation Instructions & Local Set Up
To run the Jupyter Notebook locally, you will need to have Python and the following libraries installed.
- ```pandas```
- ```numpy```
- ```matplotlib```
- ```seaborn```
- ```scikit-learn```
- ```tensorflow```
- ```gensim```

You can install these dependencies using ```pip```:
```
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow gensim
```
After installing the required libraries, you can download the project files, including the Jupyter Notebook and the bookReviewsData.csv dataset. 
**IMPORTANT** - The notebook assumes the datasets are located in a ```data``` subfolder within the project directory.
# Methodology
The project follows a standard machine learning methodology:
1.**Data Selection**: The book review dataset (bookReviewsData.csv) was chosen for this binary classification problem.
2.**Exploratory Data Analysis (EDA)**: The data was inspected for missing values, duplicates, and class imbalance. Duplicate reviews were removed, and the data was found to be fairly balanced, with no missing values.
3. **Data Preprocessing:**
   1. The text data in the 'Review' column was preprocessed using gensim.utils.simple_preprocess to convert the text to lowercase and tokenize it.
   2. The preprocessed text was then converted into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer from sklearn.feature_extraction.text.TfidfVectorizer.
4. **Model Selection and Training:**
   1. Logistic Regression: A Logistic Regression model was trained on the TF-IDF vectorized data.
   2. Neural Network: A sequential Neural Network model was constructed with multiple dense layers and dropout layers to prevent overfitting.
5. **Model Evaluation and Optimization:**
   1. Logistic Regression: The model was evaluated using accuracy and AUC (Area Under the Curve) scores. Hyperparameter tuning was performed by adjusting the C parameter.
   2. Neural Network: The model was evaluated using accuracy and loss on a validation set during training. Optimization involved experimenting with the number of epochs, units in the dense layers, and the inclusion of dropout layers.

The performance of both models was compared to determine the best-performing one.
## Sample Datasets
The project uses the following dataset:
- **bookReviewsData.csv**: This dataset contains book reviews and a corresponding boolean label indicating whether the review is positive or not.
## Jupyter/Colab Notebooks
The core of this project is documented in the Lab 8: Define and Solve an ML Problem of Your Choosing.ipynb Jupyter Notebook. It contains all the code and commentary for the project.
# Results
The performance of the two models was evaluated on the test set.
## Logistic Regression Model:
- AUC: 0.88997
- Accuracy: 0.81156
## Neural Network Model:
- Accuracy: 0.79443

# Key Findings
- The dataset was relatively clean, with no missing values. Duplicates were found and removed.
- The data exhibited a balanced distribution between positive and negative reviews, eliminating the need for techniques to address class imbalance.
- The TF-IDF vectorizer proved to be an effective method for converting text data into a format suitable for both models.
- The Logistic Regression model, with a tuned C parameter, achieved a slightly higher accuracy and a strong AUC score, indicating good performance in distinguishing between positive and negative reviews.
- The Neural Network, while also performing well, achieved a slightly lower accuracy on the test set, even after hyperparameter tuning. The dropout layers helped to regularize the model and bring the validation accuracy closer to the training accuracy, but the overall performance was not better than the simpler Logistic Regression model.

# Visualizations
The notebook includes the following visualizations:
- **Count plot:** A bar chart showing the distribution of positive vs. negative reviews to visualize class balance.
- **Loss & Accuracy Plots:** Line graphs illustrating the training and validation loss and accuracy over epochs for the Neural Network model.

# What's Next - Next Steps
- Explore other NLP techniques: Experiment with different text preprocessing methods, such as lemmatization or stemming, to see if they improve model performance.
- Try different vectorization methods: Investigate other feature extraction techniques like word2vec or GloVe embeddings.
- Implement more complex models: Test more advanced models like Recurrent Neural Networks (RNNs) or Transformers, which are often well-suited for sequence data like text.
- Perform more extensive hyperparameter tuning: Use techniques like grid search or randomized search to systematically find the optimal hyperparameters for both models.
- Conduct error analysis: Examine the misclassified reviews to identify patterns and potential reasons for the model's errors. This could provide insights for further model improvement.
