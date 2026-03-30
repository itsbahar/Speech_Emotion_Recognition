# Speech Emotion Recognition with Wav2Vec2

This project builds a **Speech Emotion Recognition (SER)** pipeline using **PyTorch** and **Hugging Face Transformers**. It fine-tunes **Facebook's Wav2Vec2** model on the **Toronto Emotional Speech Set (TESS)** dataset to classify speech into emotional categories.

## Features

- Downloads and prepares the **TESS** dataset using `kagglehub`
- Performs basic **exploratory data analysis** on audio samples
- Visualizes:
  - waveform plots
  - spectrograms
- Preprocesses audio with **Wav2Vec2Processor**
- Fine-tunes **Wav2Vec2ForSequenceClassification** for 7-class emotion classification
- Evaluates the model using:
  - accuracy
  - precision
  - recall
  - F1-score
- Runs random sample predictions on the test set

## Dataset

This notebook uses the **Toronto Emotional Speech Set (TESS)** dataset from Kaggle:

- Dataset: `ejlok1/toronto-emotional-speech-set-tess`

The dataset contains labeled speech samples for the following emotions:

- angry
- disgust
- fear
- happy
- neutral
- pleasant surprise
- sad

## Model

The model used is:

- `facebook/wav2vec2-base`

It is adapted for sequence classification with **7 output labels**.


