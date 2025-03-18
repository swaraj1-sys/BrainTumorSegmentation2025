# Brain Tumor Segmentation using ResUNet

## Overview
This project focuses on brain tumor segmentation using the **ResUNet** deep learning model. The dataset was sourced from **Kaggle**, uploaded to **Google Drive**, and the entire model was developed and trained on **Google Colab**. The aim of this project is to accurately segment tumors from brain MRI scans using a robust deep learning approach.

## Dataset
The dataset was obtained from Kaggle and consists of MRI images along with their corresponding segmentation masks. The images were uploaded to Google Drive for easy access in Google Colab.

## Technologies Used
- **Google Colab** – For model training and experimentation
- **Google Drive** – For dataset storage
- **Python** – Programming language
- **TensorFlow & Keras** – For deep learning implementation
- **OpenCV & NumPy** – For image processing
- **Matplotlib & Seaborn** – For data visualization

## Model Architecture
- The model is based on **ResUNet**, a modified U-Net architecture incorporating **Residual Blocks** to improve feature propagation and learning stability.
- Key layers include:
  - Convolutional layers with ReLU activation
  - Batch normalization for stability
  - Residual connections to prevent vanishing gradients
  - Up-sampling layers for precise segmentation

## Steps to Run the Project
1. **Clone the repository** (if hosted on GitHub):
   ```bash
   git clone https://github.com/yourusername/brain-tumor-segmentation.git
   ```
2. **Upload the dataset to Google Drive.**
3. **Open the Colab Notebook** and mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. **Install required dependencies:**
   ```python
   !pip install tensorflow opencv-python numpy matplotlib seaborn
   ```
5. **Load the dataset from Google Drive and preprocess it.**
6. **Train the ResUNet model** using the dataset.
7. **Evaluate the model** on test images.
8. **Visualize segmentation results.**

## Results
- The model successfully segments brain tumors from MRI images.
- Achieved satisfactory performance metrics such as **Dice Coefficient** and **IoU (Intersection over Union)**.
- The results are visualized using **Matplotlib** to compare ground truth vs. predicted segmentations.

## Future Improvements
- Fine-tuning hyperparameters for better accuracy.
- Using data augmentation to improve generalization.
- Experimenting with other deep learning architectures such as **U-Net++, Attention U-Net, and Transformer-based models**.

## Author
**Swaraj Pandey**

## Acknowledgments
- Thanks to **Kaggle** for providing the dataset.
- Inspired by various research papers on medical image segmentation.

---
Feel free to contribute, raise issues, or suggest improvements!

