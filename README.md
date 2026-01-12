# Automated Eye Disease Diagnosis using a Hybrid Attention-CNN

This project presents a data science research project focused on the automated classification of common eye diseases using a novel deep learning model, the Hybrid Attention-CNN (HA-CNN). The goal of this project is to provide a fast, reliable, and objective framework for the early detection of Diabetic Retinopathy (DR), Age-Related Macular Degeneration (AMD), and Glaucoma.

## Features

- **Novel HA-CNN Architecture:** A custom deep learning model that combines a pre-trained ResNet50 backbone with a self-attention mechanism for improved accuracy and interpretability.
- **Comprehensive Literature Review:** A review of over 25 SCI/Scopus references, providing a strong foundation for the research.
- **Python Implementation:** A full implementation of the solution in Python, including data preprocessing, model definition, and a training script.
- **Case Study and Draft Paper:** Detailed documentation of the research, methodology, and findings.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    ```
2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To train the model, run the `train.py` script:

```bash
python3 src/train.py
```

The script will:
1.  Load and preprocess the data from the `data/` directory.
2.  Train the HA-CNN model using 5-fold cross-validation.
3.  Generate a `classification_report.txt` and a `confusion_matrix.png` to evaluate the model's performance.

## Project Structure

```
.
├── data/
│   ├── amd/
│   ├── dr/
│   ├── glaucoma/
│   └── normal/
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── train.py
├── literature_review.md
├── comparative_analysis.md
├── case_study.md
├── draft_paper.md
└── README.md
```

## Deliverables

- **[Literature Review](literature_review.md):** A comprehensive review of the relevant literature.
- **[Comparative Analysis](comparative_analysis.md):** A comparative analysis of the proposed algorithm with existing methods.
- **[Case Study](case_study.md):** A detailed case study of the project.
- **[Draft Paper](draft_paper.md):** A draft paper summarizing the research and findings.
