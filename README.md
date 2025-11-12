# Fake News Detection using BERT-XGBoost Hybrid Model

A hybrid deep learning approach for fake news classification that combines fine-tuned DistilBERT embeddings with XGBoost classification. This project implements explainable AI (XAI) analysis using SHAP values to provide model interpretability.

## 📋 Overview

This project implements a two-stage approach for fake news detection:
1. **Stage 1**: Fine-tune DistilBERT model on the WELFake dataset for sequence classification
2. **Stage 2**: Extract contextual embeddings from the fine-tuned model and use them as features for XGBoost classification

The hybrid approach leverages the semantic understanding of BERT with the robust classification capabilities of XGBoost, achieving state-of-the-art performance on fake news detection.

## ✨ Features

- **Fine-tuned DistilBERT**: Efficient BERT-based model fine-tuned on fake news classification task
- **XGBoost Classifier**: Gradient boosting classifier on high-dimensional BERT embeddings
- **Explainable AI**: SHAP (SHapley Additive exPlanations) analysis for model interpretability
- **Comprehensive Evaluation**: Detailed metrics including precision, recall, F1-score, ROC-AUC, and confusion matrices
- **Visualizations**: Training progress, feature importance, SHAP plots, and performance metrics
- **Early Stopping**: Prevents overfitting during fine-tuning

## 🎯 Model Performance

The model achieves excellent performance on the WELFake dataset:

- **Accuracy**: 99.76%
- **Precision (Real)**: 99.73%
- **Recall (Real)**: 99.77%
- **F1-Score (Real)**: 99.75%
- **Precision (Fake)**: 99.78%
- **Recall (Fake)**: 99.74%
- **F1-Score (Fake)**: 99.76%
- **ROC-AUC**: ~1.00
- **Average Precision**: ~1.00

## 📊 Dataset

The model is trained on the **WELFake Dataset**, which contains:
- **Total samples**: 71,537 news articles (after cleaning)
- **Training set**: 57,229 samples (80%)
- **Test set**: 14,308 samples (20%)
- **Classes**: Real (0) and Fake (1)
- **Features**: Title and text content

## 🏗️ Architecture

```
Text Input (Title + Text)
    ↓
DistilBERT Tokenizer
    ↓
Fine-tuned DistilBERT Model
    ↓
[CLS] Token Embeddings (768 dimensions)
    ↓
XGBoost Classifier
    ↓
Binary Classification (Real/Fake)
```

## 🚀 Installation

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended for faster training)
- Jupyter Notebook

### Dependencies

Install the required packages:

```bash
pip install transformers[torch]
pip install xgboost
pip install datasets
pip install accelerate -U
pip install scikit-learn
pip install shap
pip install pandas numpy torch tqdm seaborn matplotlib
```

Or install all at once:

```bash
pip install transformers[torch] xgboost datasets accelerate scikit-learn shap pandas numpy torch tqdm seaborn matplotlib
```

## 📖 Usage

### 1. Data Preparation

Ensure you have the WELFake dataset in CSV format. Update the file path in the notebook:

```python
df = pd.read_csv('/path/to/WELFake_Dataset.csv')
```

### 2. Run the Notebook

Open and run `bert-xgboost-fine-tuning-welfake-updated.ipynb` in Jupyter Notebook or Google Colab.

The notebook includes the following sections:
1. **Dependencies Installation**: Install required libraries
2. **Library Imports and Configuration**: Set up environment and random seeds
3. **Data Loading and Preprocessing**: Load and clean the dataset
4. **Tokenization**: Tokenize text for BERT input
5. **Fine-Tuning DistilBERT**: Train DistilBERT on fake news classification
6. **Embedding Extraction**: Extract [CLS] token embeddings
7. **XGBoost Classification**: Train XGBoost on BERT embeddings
8. **SHAP Analysis**: Generate explainability plots
9. **Model Evaluation**: Comprehensive performance metrics
10. **Visualizations**: Training progress, ROC curves, feature importance

### 3. Model Inference

After training, you can use the model for predictions:

```python
# Load fine-tuned model
finetuned_bert_model = DistilBertModel.from_pretrained("./finetuned_distilbert")

# Extract embeddings
embeddings = get_bert_embeddings(text_loader, finetuned_bert_model, device)

# Predict with XGBoost
predictions = xgb_classifier.predict(embeddings)
```

## 📈 Results

The project generates several visualizations and metrics:

### Performance Metrics
- **Confusion Matrix**: Visualization of classification performance
- **ROC Curve**: Receiver Operating Characteristic curve
- **Precision-Recall Curve**: PR curve with average precision
- **Threshold Optimization**: Optimal decision threshold analysis

### Explainability
- **SHAP Summary Plot**: Global feature importance
- **SHAP Force Plot**: Instance-level explanations
- **Feature Importance**: Top contributing embedding dimensions

### Training Progress
- **Validation Loss**: Epoch-by-epoch validation loss with early stopping indicator

All results are saved in the `Results/` directory:
- `confusion_matrix_updated.png`
- `prf1_threshold_updated.png`
- `shap_force_plot_final.png`
- `shap_summary_plot.png`
- `validation_loss_updated.png`

## 🔍 Explainability

The project uses SHAP (SHapley Additive exPlanations) to provide model interpretability:

- **Global Interpretability**: SHAP summary plots show which embedding dimensions are most important
- **Local Interpretability**: SHAP force plots explain individual predictions
- **Feature Analysis**: Correlation analysis between features and text characteristics

## 📁 Project Structure

```
fakenews-detection-bert-xgboost-hybrid/
│
├── bert-xgboost-fine-tuning-welfake-updated.ipynb  # Main notebook
├── README.md                                        # This file
├── Results/                                         # Generated results and visualizations
│   ├── confusion_matrix_updated.png
│   ├── prf1_threshold_updated.png
│   ├── shap_force_plot_final.png
│   ├── shap_summary_plot.png
│   └── validation_loss_updated.png
│
└── finetuned_distilbert/                           # Saved fine-tuned model (generated after training)
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer_config.json
```

## 🛠️ Technologies Used

- **Transformers**: Hugging Face library for BERT models
- **XGBoost**: Gradient boosting framework
- **PyTorch**: Deep learning framework
- **SHAP**: Explainable AI library
- **scikit-learn**: Machine learning utilities and metrics
- **pandas/numpy**: Data manipulation
- **matplotlib/seaborn**: Visualization

## 🎓 Key Concepts

### DistilBERT
- Lightweight, distilled version of BERT
- 6 transformer layers (vs 12 in BERT-base)
- Faster inference while maintaining performance
- Pre-trained on large text corpora

### XGBoost
- Gradient boosting framework
- Handles high-dimensional feature spaces well
- Provides feature importance metrics
- Efficient and scalable

### SHAP Values
- Unified framework for explaining model predictions
- Based on game theory (Shapley values)
- Provides both global and local explanations
- Consistent and additive feature attributions

## 🔧 Configuration

Key hyperparameters can be adjusted in the notebook:

### DistilBERT Fine-tuning
- `num_train_epochs`: 10
- `per_device_train_batch_size`: 16
- `learning_rate`: Default (5e-5)
- `max_length`: 512 tokens
- `early_stopping_patience`: 1

### XGBoost
- `n_estimators`: 200
- `learning_rate`: 0.1
- `max_depth`: 5
- `objective`: binary:logistic

## 🚧 Future Improvements

- [ ] Support for additional datasets
- [ ] Real-time prediction API
- [ ] Model deployment and serving
- [ ] Cross-validation for robust evaluation
- [ ] Hyperparameter optimization
- [ ] Support for multi-class classification
- [ ] Integration with news sources for real-time detection
- [ ] Model compression and quantization for faster inference

## 📝 Notes

- The model requires GPU for efficient training (recommended)
- Training time: ~30-60 minutes on GPU (depending on hardware)
- The fine-tuned model is saved locally after training
- Early stopping prevents overfitting during fine-tuning

## 📄 License

This project is open source and available for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or suggestions, please open an issue on the repository.

---

**Note**: This project is for educational and research purposes. Always verify news from reliable sources and use critical thinking when evaluating information.
