# Custom Optical Character Recognition (OCR) Model

This project demonstrates the process of creating a custom Optical Character Recognition (OCR) model using PyTorch. The model is designed to recognize old-printed texts, specifically focusing on Cyrillic characters. The implementation includes data loading, preprocessing, and model training using a simple Convolutional Neural Network (CNN).

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Code Overview](#code-overview)


## Features

- Custom dataset class for OCR data handling.
- Simple CNN architecture for character recognition.
- Data loading from Excel files containing pixel values and labels.
- Training loop with loss calculation and optimization.


## Usage

To train the OCR model, you need to prepare an Excel file containing two columns: Labels and PixelValues. The Labels column should contain the Cyrillic characters, and the PixelValues column should contain the pixel representations of the images as strings.


## How It Works

1. **Data Loading**: The load_data function reads an Excel file and extracts labels and pixel values. It converts pixel values from string format to NumPy arrays and maps characters to numerical indices.

2. **Dataset Class**: The OCRDataset class inherits from PyTorch's Dataset and handles the conversion of pixel values to tensors for model input.

3. **Model Definition**: The SimpleOCRModel class defines a CNN architecture with two convolutional layers followed by fully connected layers.

4. **Training**: The train_model function splits the dataset into training and testing sets, initializes the model, loss function, and optimizer, and executes the training loop.

## Code Overview

### Key Components

- **OCRDataset Class**: Handles data loading and preprocessing.
python
  class OCRDataset(Dataset):
      def __init__(self, labels, pixel_values):
          self.labels = labels
          self.pixel_values = pixel_values

      def __len__(self):
          return len(self.labels)

      def __getitem__(self, idx):
          image = self.pixel_values[idx]
          label = self.labels[idx]
          image = torch.tensor(image, dtype=torch.float).view(1, 28, 28) / 255.0  # Normalize to [0, 1]
          return image, label

- **SimpleOCRModel Class**: Defines the CNN architecture.
python
  class SimpleOCRModel(nn.Module):
      def __init__(self, num_classes):
          super(SimpleOCRModel, self).__init__()
          self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
          self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
          self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
          self.fc1 = nn.Linear(64 * 7 * 7, 256)
          self.fc2 = nn.Linear(256, num_classes)

      def forward(self, x):
          x = self.pool(F.relu(self.conv1(x)))
  x = self.pool(F.relu(self.conv2(x)))
          x = x.view(-1, 64 * 7 * 7)
          x = F.relu(self.fc1(x))
          x = self.fc2(x)
          return x
  
  
  


