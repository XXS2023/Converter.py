import pandas as pd
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image, ImageFilter
import torchvision.transforms as transforms


# Define the OCR Dataset class
class OCRDataset(Dataset):
    def __init__(self, labels, pixel_values):
        self.labels = labels
        self.pixel_values = pixel_values


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.pixel_values[idx]
        label = self.labels[idx]
        # Convert pixel values to tensor and reshape to 1x28x28
        image = torch.tensor(image, dtype=torch.float).view(1, 28, 28) / 255.0  # Normalize to [0, 1]
        return image, label


# Simple CNN Model for OCR
class SimpleOCRModel(nn.Module):
    def __init__(self, num_classes):
        super(SimpleOCRModel, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)  # Adjusted based on pooling
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load and preprocess data
def load_data(excel_file):
    df = pd.read_excel(excel_file)

    # Extract labels and pixel values from the DataFrame
    labels = df['Labels'].tolist()  # Column with Cyrillic labels
    pixel_values = df['PixelValues'].apply(lambda x: np.fromstring(x.strip('[]'),
                                                                   sep=',')).tolist()  # Convert string representation of list to actual list

    # Create a mapping from characters to numerical indices
    unique_chars = sorted(set(labels))
    global char_to_idx
    char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
    global idx_to_char
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    # Convert labels to numerical indices
    numerical_labels = [char_to_idx[label] for label in labels]

    return numerical_labels, pixel_values, char_to_idx


# Main function to train the model
def train_model(excel_file):

    labels, pixel_values, char_to_idx = load_data(excel_file)
    # Split into train and test sets (80% train, 20% test)
    train_labels, test_labels, train_pixel_values, test_pixel_values = train_test_split(labels, pixel_values,
                                                                                        test_size=0.2)
    # Create datasets and dataloaders
    train_dataset = OCRDataset(train_labels, train_pixel_values)
    test_dataset = OCRDataset(test_labels, test_pixel_values)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    global test_loader
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Initialize model, loss function and optimizer
    model = SimpleOCRModel(num_classes=len(char_to_idx))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):  # Number of epochs
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Training complete.")
    torch.save(model.state_dict(), 'ocr_model.pth')

    # Evaluation phase
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with (torch.no_grad()):  # Disable gradient calculation for evaluation
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted_indices = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted_indices == labels).sum().item()

    accuracy = correct / total * 100
    print(f'Accuracy on test set: {accuracy:.2f}')


def imageprepare(argv):

    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.LANCZOS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.LANCZOS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    tv = list(newImage.getdata())  # get pixel values
    return tv

def preprocess_and_recognize_word(model, folder_path):
    recognized_chars = []
    for image_file in os.listdir(folder_path):
        if image_file.endswith(".jpg"):
            image_path = os.path.join(folder_path, image_file)
            resized_image = imageprepare(image_path)
            images_data = np.array(resized_image)
        char_image_tensor = torch.tensor(images_data / 255.0, dtype=torch.float).view(1, 1, 28, 28)

        with torch.no_grad():
            output = model(char_image_tensor)
            _, predicted_index = torch.max(output, 1)
            recognized_chars.append(idx_to_char[predicted_index.item()])
    recognized_word = ''.join(recognized_chars)
    return recognized_word

# Example usage
excel_file_path = r'C:/Users/vorob/OneDrive/Рабочий стол/Alpha/training_dataset.xlsx'
train_model(excel_file_path)

    # Load the trained model for recognition
model = SimpleOCRModel(num_classes=len(char_to_idx))
model.load_state_dict(torch.load('ocr_model.pth'))

    # Evaluate the model on the test set
evaluate_model(model, test_loader)
  # Example of recognizing a word from a scanned image
path = 'pronaun_characters'
recognized_word = preprocess_and_recognize_word(model, path)
print(f'Recognized Word: {recognized_word}')







