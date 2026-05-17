# Deep-Learning-Traffic-Sign-Classifier
Traffic sign classification using CNN and PyTorch on the GTSRB dataset with model enhancements and performance comparison.
# Traffic Sign Classification using CNN (GTSRB)

## Project Description

This project implements a Deep Learning approach for **Traffic Sign Classification** using Convolutional Neural Networks (CNN) and the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

The objective is to classify traffic sign images into one of **43 classes** such as speed limits, stop signs, warning signs, and road instructions.

The project includes:

- Data preprocessing
- Dataset visualization
- CNN model implementation
- Model enhancement using Batch Normalization and Dropout
- Multiple experiments
- Performance evaluation
- Results comparison and visualization
- Testing on custom images

---

## Dataset

Dataset used:

German Traffic Sign Recognition Benchmark (GTSRB)

Dataset source:

https://pytorch.org/vision/main/generated/torchvision.datasets.GTSRB.html

Number of classes: **43**

Dataset split:

- Training Set
- Validation Set
- Testing Set

---

## Data Preprocessing

The following preprocessing steps were applied:

- Resize images to **32×32**
- Convert images to tensors
- Normalize image values
- Create train/validation split
- Batch loading using DataLoader

Transformation pipeline:

```python
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,0.5,0.5),
        (0.5,0.5,0.5)
    )
])
```

---

## Model Architecture

### Model A — Baseline CNN

Architecture:

- Conv2D (3 → 32)
- ReLU
- MaxPooling
- Conv2D (32 → 64)
- ReLU
- MaxPooling
- Flatten
- Fully Connected Layer
- Output Layer (43 classes)

---

### Model B — Enhanced CNN

Additional improvements:

- Batch Normalization
- Dropout
- Lower Learning Rate

Architecture:

- Conv2D
- BatchNorm
- ReLU
- MaxPooling
- Conv2D
- BatchNorm
- ReLU
- MaxPooling
- Dropout
- Fully Connected
- Dropout
- Output Layer

---

## Hyperparameters

| Parameter | Value |
|------------|--------|
| Batch Size | 64 |
| Learning Rate (Model A) | 0.001 |
| Learning Rate (Model B) | 0.0007 |
| Epochs | 10 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |

---

## Experiments

Two experiments were conducted:

### Experiment A — Baseline CNN

Features:

- Basic CNN architecture
- Adam optimizer
- Learning rate = 0.001

---

### Experiment B — Enhanced CNN

Features:

- Batch Normalization
- Dropout
- Lower learning rate

---

## Training Results

### Model A

Test Accuracy:

**86.93%**

Test Loss:

**0.6580**

---

### Model B

Test Accuracy:

**94.77%**

Test Loss:

**0.1959**

---

## Results Comparison

| Model | Accuracy | Loss |
|---------|----------|-------|
| Baseline CNN | 86.93% | 0.6580 |
| Enhanced CNN | 94.77% | 0.1959 |

---

## Analysis

The enhanced model achieved significantly better performance.

Reasons:

- Batch Normalization stabilized training
- Dropout reduced overfitting
- Lower learning rate improved convergence
- Better generalization on unseen data

Improvement:

Accuracy increased from:

86.93% → 94.77%

Approximate improvement:

+7.85%

---

## Visualizations

The project includes:

- Training Loss vs Validation Loss
- Training Accuracy vs Validation Accuracy
- Experiment comparison plots
- Accuracy comparison bar chart

---

## Testing on Custom Images

The trained model was tested on external traffic sign images:

Example predictions:

- Traffic sign image
- Stop sign image

Prediction is generated using:

```python
model.eval()

with torch.no_grad():
    output = model(img)
    prediction = torch.argmax(output,1)
```

---

## Project Structure

```text
Traffic-Sign-Classification/
│
├── notebook.ipynb
├── README.md
└── results/
 
```

---

## How to Run

### Clone repository

```bash
git clone YOUR_REPOSITORY_LINK
```

### Install dependencies

```bash
pip install torch torchvision matplotlib numpy pandas pillow
```

### Run notebook

Open Jupyter Notebook:

```bash
jupyter notebook
```

Run all cells sequentially.

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib
- Pandas
- PIL

---

## Conclusion

This project demonstrates how Convolutional Neural Networks can effectively classify traffic signs.

The enhanced CNN model using Batch Normalization and Dropout achieved the best performance with:

**94.77% test accuracy**

showing substantial improvement over the baseline model.
