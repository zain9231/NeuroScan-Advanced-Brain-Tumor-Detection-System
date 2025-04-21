# NeuroScan Advanced Brain Tumor Detection System

NeuroScan is a Final Year Project (FYP) focused on the development of an advanced system for brain tumor detection using machine learning and deep learning techniques. The project addresses the critical need for accurate and efficient diagnostic tools in medical imaging, aiming to support medical professionals in the timely identification of brain tumors in MRI and CT scans.

## Project Domain  Category

Image Processing  Deep Learning

## Abstract

This project presents NeuroScan, an innovative system designed to enhance the accuracy and efficiency of brain tumor detection through the application of advanced machine learning and medical imaging techniques. Given the increasing global incidence of brain tumors, early and precise diagnosis is paramount for effective intervention and treatment planning. NeuroScan provides a robust and reliable automated detection system leveraging deep learning algorithms, specifically a YOLOv8-based architecture, trained on comprehensive medical image datasets (MRI and CT scans). The system is engineered to identify patterns and anomalies indicative of brain tumors with high precision and sensitivity, thereby assisting medical professionals in their diagnostic workflows.

## Features

NeuroScan incorporates a suite of functionalities to facilitate the brain tumor detection process

 Automated Brain Tumor Detection Implementation of deep learning models (including YOLOv8) for the identification of potential brain tumors in medical images.
 Image Preprocessing Pipeline Integration of essential preprocessing steps including image normalization, noise reduction, skull stripping, and artifact removal to optimize image quality for analysis.
 Data Augmentation Framework Application of various data augmentation techniques (rotation, translation, scaling, flipping, elastic deformation, intensity adjustment, noise injection, shearing, random cropping) to increase dataset diversity and model robustness.
 Real-time Detection Capability for real-time or near-real-time tumor detection with an adjustable confidence threshold to accommodate diagnostic requirements.
 Graphical User Interface (GUI) Development of an intuitive interface for image input, visualization of detection results, display of diagnostic information, and interactive control over system parameters.
 Batch Processing Support for processing multiple images efficiently in a batch mode.
 Results Management Functionality to save and export processed images with overlaid detection results.

## Getting Started

These instructions will guide you through setting up and running the NeuroScan system.

### Prerequisites

 Python 3.8 or higher
 Fundamental knowledge of Artificial Intelligence, Machine Learning, Image Processing, and Computer Vision concepts.

### Installation

1.  (Assuming the codebase is hosted on GitHub) Clone the project repository
    ```bash
    git clone repository_url
    cd NeuroScan
    ```
2.  Install the required Python libraries as specified in the `requirements.txt` file
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  Execute the primary application script to launch the GUI
    ```bash
    python main.py
    ```
2.  Utilize the interface controls (buttons or menus) to select individual images for detection or specify dataset folders for batch processing, preprocessing, or augmentation.
3.  Optional preprocessing and augmentation steps can be configured and applied via their respective panels in the main window.
4.  Initiate the tumor detection process through the dedicated detection window.
5.  Review the processed images with visual indicators of detected tumors and consult the information panel for detailed diagnostic data.
6.  Adjust the confidence threshold to observe its effect on the detection sensitivity.
7.  Navigate between original and processed images and utilize zoom functionalities for detailed examination.
8.  Save processed images with detection overlays as needed.
9.  Employ the batch processing feature to analyze multiple images sequentially.

## Dataset

The dataset employed for the development and evaluation of NeuroScan comprises Brain MRI and CT images designated for brain tumor detection.

Dataset Information
The dataset includes images of varying resolutions and qualities. Dataset was provided by my supervisor.

## Data Augmentation

Data augmentation was implemented to address limitations associated with dataset size and to enhance model generalization capabilities by increasing the diversity of the training data. The framework supports a comprehensive suite of augmentation techniques applied to the dataset, including geometric transformations and intensity adjustments.

## Data Preprocessing

Prior to model ingestion, images undergo a series of preprocessing steps to standardize input and mitigate potential confounding factors. The preprocessing pipeline includes

1.  Image Normalization Standardizing pixel intensity values across the dataset.
2.  Noise Reduction Application of filtering techniques (e.g., Gaussian or median filters) to minimize image noise.
3.  Skull Stripping Isolation of brain tissue by removing non-cerebral structures.
4.  Artifact Removal Identification and mitigation of imaging artifacts.

## Data Split

The dataset was partitioned to facilitate model training and objective performance evaluation

1.  Training Set 70% of the dataset allocated for model training.
2.  Testing Set 30% of the dataset reserved for evaluating the trained model's performance on unseen data.

## Neural Network Architecture

The core of the NeuroScan system is a deep learning model based on the YOLOv8 architecture. This architecture was selected following an investigation into various deep learning models suitable for object detection in medical images. The implementation is tailored for the specific task of brain tumor detection within MRI and CT modalities.

## Model Training

The model was trained using the prepared dataset with the following parameters

 Optimizer Adam
 Loss Function Categorical cross-entropy
 Batch Size 16
 Epochs 50

Regularization techniques, including dropout and early stopping, were applied to prevent overfitting and enhance model performance.

Markdown

## Results

The performance of the NeuroScan model was evaluated using standard classification metrics, providing insights into its efficacy in brain tumor detection.

| Metric            | Value |
| :---------------- | :---- |
| Accuracy          | 85%   |
| Precision         | 83%   |
| Recall            | 87%   |
| F1-Score          | 0.85  |
| Specificity       | 82%   |
| AUC (ROC Curve)   | 0.83  |

**Confusion Matrix:**
The confusion matrix analysis revealed detailed performance across the four classes. The model demonstrated strong performance in correctly classifying instances of Class 1 (1626 instances) but exhibited limitations in correctly identifying instances of Class 3 (0 correct predictions), with a notable number of misclassifications involving Class 3.

**ROC Curve:**
The Receiver Operating Characteristic (ROC) curve visualizes the trade-off between the True Positive Rate and the False Positive Rate at varying thresholds. The curve's characteristics and the AUC value (0.83 or 0.88) indicate that the model possesses a significant capacity to discriminate between positive and negative cases, exceeding random chance performance.

**F1-Confidence Curve:**
The F1-Confidence curve illustrates the relationship between the F1 score and the classification confidence threshold. The plot indicates that the peak overall F1 score of 0.83 is achieved at a confidence threshold of approximately 0.446, suggesting this threshold optimizes the balance between precision and recall.

## Credits

This project was conducted as a Final Year Project by myself under the supervision of Mr. Umair Ali.

## Final Notes

What's in the Files

 The core project code is contained within the Python scripts (`.py` files) listed in the Project Structure.
 The trained weights for the YOLOv8 detection model are used by the application. The model loading is handled within the `model.py` file using the `ultralytics` library.
 The original dataset should be organized in a folder you select. The application will create 'preprocessed' and 'augmented' subfolders within the selected dataset folder when you perform those operations.

Contributions are welcome!

Thank you for your interest in NeuroScan!
