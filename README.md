# Next Word Prediction (LSTM)

A deep learning project that predicts the next word for a given input sentence using an LSTM (Long Short-Term Memory) model trained on a quotes dataset, served through an interactive Streamlit web interface.

---

## Tech Stack & Tools

- **Language:** Python 3.11
- **Deep Learning:** TensorFlow / Keras (Embedding, LSTM, Dense layers)
- **Web Interface:** Streamlit
- **Data Handling:** Pandas, NumPy
- **Serialization:** Pickle (saving/loading tokenizer and sequence length)
- **Notebooks:** Jupyter Notebook / Google Colab (for model training and experimentation)

---

## Project Structure

```text
├── app.py                  # Streamlit web application
├── lstm_model (1).h5       # Pre-trained LSTM model weights
├── tokenizer.pkl           # Saved Keras tokenizer vocabulary
├── max_len.pkl             # Maximum sequence length used for padding
├── qoute_dataset.csv       # Training quotes dataset
├── codefile.ipynb          # Model training and data exploration notebook
├── RNNimplementation.ipynb # Model architecture experimentation notebook
├── requirements.txt        # Project dependencies
└── README.md
```

---

## How It Works

1. **Training & Preprocessing:** The dataset (`qoute_dataset.csv`) is tokenized into n-gram sequences. Sequences are padded to uniform length (`max_len`).
2. **Model:** An LSTM architecture learns word dependencies and probability distributions across the vocabulary.
3. **Inference:** In `app.py`, the user's input string is converted into tokens using `tokenizer.pkl`, padded to `max_len - 1`, and fed into `lstm_model (1).h5`. The word with the highest predicted probability is returned.

---

## Getting Started

### 1. Clone & Navigate
```bash
git clone <repo-url>
cd Next_word_prediction-main
```

### 2. Set Up a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Web App
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## Retraining the Model

To modify the architecture or retrain on new data:
1. Open [codefile.ipynb](codefile.ipynb) or [RNNimplementation.ipynb](RNNimplementation.ipynb) in Jupyter / Colab.
2. Run through the data preprocessing, tokenization, and training cells.
3. Export the updated `model.h5`, `tokenizer.pkl`, and `max_len.pkl` into the root directory.
