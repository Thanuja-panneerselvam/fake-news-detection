# Fake News Detection

This project implements a simple fake news detection system using natural language processing (NLP) techniques and machine learning. The model is trained to classify news articles as either fake or real based on their textual content.

## Features

- Basic text preprocessing to clean and prepare the data.
- Synthetic dataset creation for training and testing the model.
- Utilizes TF-IDF vectorization for feature extraction.
- Implements a Logistic Regression model for classification.
- Evaluates model performance using accuracy and classification report metrics.
- Provides a function to predict whether a given text is fake news or real news.

## Prerequisites

- Python 3.x
- Required Python packages:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `nltk`

You can install the required packages using pip:

```bash
pip install pandas numpy scikit-learn nltk
```

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/fake-news-detection.git
   cd fake-news-detection
   ```

2. Ensure you have the necessary NLTK resources downloaded. The script will automatically download the required resources when run.

## Usage

1. Run the script:

   ```bash
   python fake_news_detector.py
   ```

2. The script will train the model on a synthetic dataset and evaluate its performance. It will then test the model with some example texts and print the results.

3. The output will include:
   - Model accuracy.
   - A classification report detailing precision, recall, and F1-score.
   - Predictions for the test texts, indicating whether they are classified as fake news or real news along with the confidence level.

## Example Output

The script will print the model's accuracy, a classification report, and the results of the test texts, showing whether each text is classified as fake news or real news along with the confidence score.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [NLTK](https://www.nltk.org/) for natural language processing tools.
- [Scikit-learn](https://scikit-learn.org/) for machine learning algorithms.
