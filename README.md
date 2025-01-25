# Sentiment Analysis Pipeline with BERT

## Introduction
This repository contains a sentiment analysis pipeline developed using the **BERT** (Bidirectional Encoder Representations from Transformers) model for classifying Amazon product reviews into three sentiment categories: **Positive**, **Negative**, and **Neutral**. The dataset used is from the **Amazon Reviews** dataset, specifically the "All Beauty" category, containing raw customer reviews. The goal of this project is to train a model capable of determining the sentiment of a review based on the rating given (from 1 to 5 stars).

The project showcases:
1. **Sentiment Classification**: Categorizing reviews as positive, neutral, or negative based on their rating.
2. **Data Preprocessing**: Tokenizing the raw text data and mapping ratings to sentiment labels.
3. **Model Training**: Fine-tuning the BERT model for multi-class classification.
4. **Evaluation**: Evaluating model performance using metrics such as accuracy, precision, recall, and F1 score.

---

## Tools and Libraries Used

### Programming Language
- **Python 3.x**: The primary language used for data processing, model training, and evaluation.

### Libraries and Frameworks
- **NumPy**: For numerical operations and handling arrays.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For data visualization, such as plotting graphs to understand the distribution of data.
- **Hugging Face Transformers**: For working with pre-trained models like BERT.
- **Scikit-learn**: For model evaluation, metrics calculation, and data splitting.
- **Datasets**: For loading the Amazon Reviews dataset from Hugging Face's `datasets` library.
- **Torch**: For training the BERT model and running the computations.

---

## Dataset Description
The repository includes the **Amazon Reviews 2023** dataset, specifically from the **All Beauty** category, available on Hugging Face. The dataset contains product reviews with the following features:

### Features:
- **rating**: The rating given by the customer (ranging from 1 to 5 stars).
- **title**: The title of the review provided by the customer.
- **text**: The actual content of the review left by the customer.
- **images**: Any associated images with the review (if applicable).
- **asin**: The Amazon Standard Identification Number, a unique identifier for the product.
- **parent_asin**: The parent ASIN for product variants (if applicable).
- **user_id**: The unique identifier for the customer who left the review.
- **timestamp**: The timestamp when the review was submitted.
- **helpful_vote**: The number of people who found the review helpful.
- **verified_purchase**: Whether the purchase was verified (True/False).

### Sentiment Labels:
The sentiment labels were created by me based on the rating scale:
- **Positive** (Label 2): Ratings of **4–5 stars**.
- **Neutral** (Label 1): Ratings of **3 stars**.
- **Negative** (Label 0): Ratings of **1–2 stars**.

### Dataset Size:
- **Number of Reviews**: The dataset contains a large number of reviews for beauty products.
- **Data Splits**: The data is split into training, validation, and test sets for model evaluation.

---

## Key Features and Workflow

### 1. Data Preprocessing
- **Loading the Dataset**: The raw dataset is loaded from the Hugging Face `datasets` library.
- **Mapping Ratings to Sentiment Labels**: A custom function `encode_labels` is used to convert the numerical ratings (1–5 stars) into sentiment labels (Negative, Neutral, Positive).
- **Splitting the Data**: The dataset is split into training, validation, and test sets using **train_test_split** from **scikit-learn**.

### 2. Tokenization
- **BERT Tokenizer**: The **BERT tokenizer** is used to preprocess and tokenize the reviews. This includes padding and truncating reviews to a fixed length for compatibility with BERT's input format.
- **Tokenized Dataset**: After tokenization, the dataset is transformed into a format suitable for input to the BERT model.

### 3. Model Training
- **BERT for Sequence Classification**: The pre-trained BERT model is fine-tuned for multi-class classification, where it learns to predict one of the three sentiment classes (Negative, Neutral, Positive).
- **Training Arguments**: The training is configured using the **TrainingArguments** class, with hyperparameters such as learning rate, batch size, number of epochs, and logging frequency.
- **Trainer**: The **Trainer** class from Hugging Face simplifies the training loop and model evaluation.

### 4. Model Evaluation
- **Metrics**: The model's performance is evaluated using multiple metrics:
  - **Accuracy**: Proportion of correct predictions.
  - **Precision**: Proportion of positive predictions that are correct.
  - **Recall**: Proportion of actual positives that are correctly identified.
  - **F1 Score**: Harmonic mean of precision and recall.
- **Test Set Evaluation**: The model is evaluated on the test set, and the results are displayed.

### 5. Prediction Flow
1. **Input**: Reviews are tokenized and passed into the trained BERT model.
2. **Prediction**: The model predicts the sentiment class for each review.
3. **Output**: The predicted sentiment is returned, either as Positive, Neutral, or Negative.

---

## Future Improvements
- **Fine-tuning with Custom Models**: Experimenting with other pre-trained models such as **RoBERTa** or **DistilBERT** for potential improvements in performance.
- **Hyperparameter Tuning**: Use techniques like **GridSearchCV** or **RandomizedSearchCV** to fine-tune hyperparameters and improve the model.
- **Deployment**: Develop an API or web application for real-time sentiment analysis predictions.

---

## Conclusion
This project demonstrates the power of **BERT** for sentiment analysis and showcases how pre-trained transformer models can be fine-tuned for specific tasks like review sentiment classification. The pipeline provides a robust solution for analyzing customer feedback in the form of product reviews and can be extended to other domains or datasets.

---

## Contact
For questions, suggestions, or feedback, please reach out via:
- **Email**: [pharezayodele6@gmail.com]
- **LinkedIn**: [www.linkedin.com/in/pharez-ayodele-73b13021b]
