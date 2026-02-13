
# Contextual Bandit – User Classification (Lab 3)

## Overview

This project implements a contextual learning pipeline to classify users based on contextual features and map them to recommendation arms.

The workflow includes:

* Loading and preprocessing user datasets
* Encoding categorical variables
* Training a machine learning classifier
* Evaluating performance on validation and test datasets
* Visualizing results

The system simulates a simplified contextual bandit scenario where user context determines the best category (arm).

---

## Project Structure

```
lab3-contextual-bandit/
│
├── data/
│   ├── train_users.csv
│   ├── test_users.csv
│
├── lab3_results_U20230039.ipynb
├── README.md
└── requirements.txt
```

---

## Requirements

Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn rlcmab-sampler
```

---

## Dataset

The dataset contains user context features and a label representing the target category.

Typical fields may include:

* Demographic or behavioral features
* Encoded categorical attributes
* Label (target category)

Two datasets are used:

* Training dataset
* Test dataset

---

## Methodology

### 1. Data Preprocessing

* Convert labels to consistent format
* Handle missing values
* Encode categorical features
* Split dataset into training and validation sets

---

### 2. Model Training

A classifier is trained to predict the user category from contextual features.

Steps:

* Separate features and labels
* Train model on training split
* Validate performance on validation split

---

### 3. Context to Arm Mapping

User contexts are mapped to numerical arms representing recommendation categories.

Example:

```
Entertainment → Arm 0  
Education → Arm 1  
Tech → Arm 2  
Crime → Arm 3  
```

---

### 4. Evaluation

The model is evaluated using:

* Validation accuracy
* Test accuracy

Plots are generated to visualize performance.

---

## Running the Project

Open the notebook:

```bash
jupyter notebook lab3_results_U20230039.ipynb
```

Run all cells sequentially.

---

## Results

The trained model successfully predicts user categories based on contextual features. Performance is measured using accuracy on validation and test datasets.

(Add screenshots of plots here if required for submission.)

---

## Future Improvements

Possible extensions:

* Implement full contextual bandit algorithms (UCB, Thompson Sampling)
* Hyperparameter tuning
* Neural contextual bandit models
* Online learning setup
