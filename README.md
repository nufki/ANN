# Simple Java ANN (Multi-Layer Perceptron)

This project contains a **small, self-contained implementation of a neural network**
(also called a **Multi-Layer Perceptron**, MLP) written in plain Java.

It is designed for **learning, experimenting, and understanding how neural networks work internally**.

No external libraries are used.

---

## What is implemented?

The `ANN` class implements:

- One **input layer**
- One **hidden layer**
- One **output layer**
- **Sigmoid** activation functions
- **Backpropagation** learning
- **Stochastic Gradient Descent (SGD)**
- **Mean Squared Error (MSE)** loss

---

## Network Structure

The network structure is defined when creating an instance:
```java
ANN ann = new ANN(inputs, hiddenNeurons, outputs);
```

Meaning:

- `inputs` → number of values in one input sample
- `hiddenNeurons` → number of neurons in the hidden layer
- `outputs` → number of values produced by the network

Example:
```java
ANN ann = new ANN(2, 3, 1);
```

---

## Core Methods

### `double[] apply(double[] input)`

Runs a **forward pass** through the network.

- Computes hidden layer activations
- Computes output layer activations
- Returns the network output

---

### `void train(double[][] X, double[][] Y, int epochs, double learningRate)`

Trains the network using supervised learning via backpropagation.

---

## Example: XOR Problem

The XOR problem is a classic neural network example:

| Input | Output |
|------|--------|
| 0, 0 | 0 |
| 0, 1 | 1 |
| 1, 0 | 1 |
| 1, 1 | 0 |

This problem **cannot be solved with a single linear layer**, which makes it a perfect test for a neural network with a hidden layer.

### Example usage
```java
double[][] X = {
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1}
};

double[][] Y = {
    {0},
    {1},
    {1},
    {0}
};

ANN ann = new ANN(2, 3, 1);
ann.train(X, Y, 50_000, 0.5);

for (double[] x : X) {
    double[] out = ann.apply(x);
    System.out.println(
        "(" + (int)x[0] + ", " + (int)x[1] + ") -> " + out[0]
    );
}
```

---

## Real-World Application: Gender Classification from Face Images

The **GenderClassifier** demonstrates the neural network applied to a practical computer vision task: classifying gender from facial photographs.

### Overview

- **Task**: Binary classification (Male vs Female)
- **Input**: 48×48 grayscale face images (2,304 pixels)
- **Output**: Probability of being female (0.0 = male, 1.0 = female)
- **Architecture**: 2,304 → 32 → 1
- **Performance**: ~80% test accuracy on Adience dataset

### Network Architecture
```java
GenderClassifier classifier = new GenderClassifier(
    hiddenNeurons: 32,
    learningRate: 0.05,
    seed: 42
);
```

The network processes each image as follows:
1. **Preprocessing**: Resize to 48×48, convert to grayscale, normalize pixels to [0,1]
2. **Forward pass**: 2,304 input neurons → 32 hidden neurons → 1 output neuron
3. **Classification**: Output > 0.5 = Female, Output ≤ 0.5 = Male

### Dataset Preparation

The classifier uses the **Adience Benchmark Gender and Age Classification** dataset from Kaggle:

https://www.kaggle.com/datasets/ttungl/adience-benchmark-gender-and-age-classification/data

#### Step 1: Download and Extract Dataset

1. Download the dataset from Kaggle
2. Unzip to the project root directory
3. Run the Python script `copyImages.py` to extract images:
```python
# Edit the script to select the desired fold
df = pd.read_csv('AdienceBenchmarkGenderAndAgeClassification/fold_0_data.txt', sep='\t')
```

This copies images to: `src/main/resources/faces_dataset/`

#### Step 2: Crop and Prepare Images

Many images contain backgrounds or multiple faces. The `FaceCropTool` (located in the `tools` package) preprocesses images:
```java
// Runs center-crop on all images and resizes to 48×48
FaceCropTool.main(args);
```

This creates the cleaned dataset at: `src/main/resources/faces_dataset_cropped/`

### Training the Model
```java
// Load preprocessed dataset
Dataset data = GenderClassifier.loadDataset("src/main/resources/faces_dataset_cropped");

// Split into train/test
Dataset[] split = GenderClassifier.trainTestSplit(data, 0.8);

// Train with early stopping
classifier.train(trainData.X, trainData.Y, epochs);
```

### Key Training Features

**Early Stopping**: Training automatically stops when test accuracy stops improving for 100 consecutive epochs, preventing overfitting.

Example output:
```
Epoch   50: Train=88.4% Test=82.1% MSE=0.0953 (Best: 82.1% @ epoch 50)
Epoch  100: Train=93.9% Test=83.7% MSE=0.0546 (Best: 83.7% @ epoch 98)
Epoch  150: Train=96.5% Test=83.2% MSE=0.0345 (Best: 83.7% @ epoch 98)

Early stopping at epoch 198 (no improvement for 100 epochs)
⚠️  Best model was at epoch 98 with 83.7% test accuracy
```

**Important**: The "Best" accuracy shown during training is the true model performance, not the final accuracy (which may be lower due to overfitting).

### Performance Characteristics

| Dataset Size | Hidden Neurons | Parameters | Test Accuracy | Training Time* |
|-------------|----------------|------------|---------------|----------------|
| 961 images  | 16             | 36,897     | ~70-75%       | ~2 min         |
| 4,091 images| 32             | 73,793     | ~80-83%       | ~5 min         |

*Approximate, varies by hardware

### Critical Design Decisions

**Why 48×48 resolution?**
- Balance between detail and computational efficiency
- 2,304 features vs 150,528 for 224×224 images
- Sufficient for facial features (eyes, nose, mouth, hair)

**Why only 32 hidden neurons?**
- Parameter-to-sample ratio should be < 25 for good generalization
- 73,793 parameters ÷ 3,272 training samples = 22.5 ratio
- Prevents overfitting on limited dataset

**Why MSE instead of Cross-Entropy?**
- Both work for binary classification
- MSE with sigmoid is mathematically equivalent after gradient cancellation
- Simpler to implement and understand

### Limitations

- **Simple architecture**: Fully-connected networks don't capture spatial patterns like CNNs
- **Small dataset**: Professional systems use 100,000+ images
- **No data augmentation**: Could improve accuracy by 5-10%
- **Single face assumption**: Doesn't handle multiple faces or face detection
- **Limited preprocessing**: No face alignment or landmark detection

### Expected Accuracy Ranges

| Accuracy | Interpretation |
|----------|---------------|
| 50-60%   | Network not learning (random guessing) |
| 60-70%   | Learning but dataset issues (babies, multiple faces) |
| 70-80%   | Good performance for a simple MLP |
| 80-85%   | Excellent for fully-connected network |
| 85-90%   | Requires CNNs or much larger dataset |
| 90-95%   | State-of-the-art (deep CNNs on massive datasets) |

---

## Disclaimer

This project is intended for **educational purposes only**.

---

## Visualizations

### XOR Problem - Decision Boundary
![xor.png](readme-images/xor.png)

### 3 Class Problem - Decision Boundary
![3-class-problem.png](readme-images/3-class-problem.png)

### 3 Class Problem with Model Overfitting
![3-class-problem-overfitting.png](readme-images/3-class-problem-overfitting.png)