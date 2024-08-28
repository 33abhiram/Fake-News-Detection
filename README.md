# COVID-19 Fake News Detection

This project focuses on detecting fake news from COVID-19-related tweets. Using a dataset with labeled tweets, I explore various machine learning models to classify whether a tweet is real or fake.

## Dataset
- **Train Data**: `FakeNews_train.csv`
- **Test Data**: `FakeNews_test.csv`
- **Columns**: 
  - `tweet`: The text of the tweet.
  - `label`: Indicates if the tweet is "real" or "fake".

## Project Structure
1. **Data Preprocessing**
   - Loading datasets and cleaning text data (removal of URLs, hashtags, mentions).
   - Tokenization, stop words removal, and lemmatization.
   - Processed data saved into new files.

2. **Model Definition**
   - Implemented three machine learning models:
     - Logistic Regression
     - Support Vector Classifier (SVC)
     - Multi-Layer Perceptron Classifier (MLP)

3. **Feature Extraction & Model Evaluation**
   - **TF-IDF Features**:
     - Extracted TF-IDF features (max_features=5000) for each example.
     - Trained and evaluated all three models.
   - **Averaged Word Embeddings**:
     - Used pre-trained word2vec model for word embeddings and averaged them to form sentence embeddings.
     - Trained and evaluated the models.
   - **Sentence Transformer Embeddings**:
     - Direct extraction of sentence embeddings using sentence transformers.
     - Trained and evaluated the models.

4. **Observations**
   - Identified the best-performing model based on evaluation metrics.
   - Provided insights by printing 10 examples from the test data with actual and predicted labels.

## References
Das, S. D., Basak, A., & Dutta, S. (2021). A heuristic-driven ensemble framework for COVID-19 fake news detection. In Combating Online Hostile Posts in Regional Languages during Emergency Situation, CONSTRAINT 2021, Collocated with AAAI 2021, February 8, 2021.

## Usage
- **Run the Notebook**: Open the provided Jupyter notebook file to execute the entire workflow.
- **Dependencies**: Ensure all required Python libraries are installed (e.g., scikit-learn, pandas, nltk, gensim, sentence-transformers).
