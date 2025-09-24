# 🎬 **IMDB Movie Review Sentiment Analysis** using BERT, Classical ML, and BiLSTM+GloVe

This project performs **sentiment analysis** on IMDB movie reviews, comparing the performance of **BERT**, **Classical Machine Learning**, and **BiLSTM + GloVe** models.

## 📊 **Key Results**

* **BERT (Base)**: **92% Accuracy**
* **Classical ML (SVM, Naive Bayes, etc.)**: **90% Accuracy**
* **BiLSTM + GloVe**: **89% Accuracy**

## 🧠 **Models Used**

1. **BERT (Base)**: Fine-tuned **BERT** model for contextualized word representations, achieving the best performance.
2. **Classical ML**: Used traditional models like **SVM**, **Naive Bayes**, and **Logistic Regression** on bag-of-words or TF-IDF features.
3. **BiLSTM + GloVe**: Bidirectional LSTM model combined with **GloVe embeddings** for capturing sequential dependencies in reviews.

## 🚀 **Getting Started**

1. **Clone the Repo**:

   ```bash
   git clone https://github.com/Mounika-github99/IMDB-Sentiment-Analysis.git
   cd IMDB-Sentiment-Analysis
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Models**:

   * For **BERT**:

     ```bash
     python train_bert.py
     ```
   * For **Classical ML**:

     ```bash
     python train_classical_ml.py
     ```
   * For **BiLSTM + GloVe**:

     ```bash
     python train_bilstm_glove.py
     ```

4. **Run Inference** on new reviews:

   ```bash
   python predict.py --model bert --text "This movie is amazing!"
   ```

## 🔍 **Model Evaluation**

* **BERT**: Best performing with **92%** accuracy, leveraging pre-trained transformer-based embeddings.
* **Classical ML**: Achieved **90%** accuracy using traditional techniques with TF-IDF features.
* **BiLSTM + GloVe**: **89%** accuracy, effectively learning sequential dependencies in text.

This format gives the key points and results in a clean, straightforward manner! Let me know if you'd like any tweaks.
