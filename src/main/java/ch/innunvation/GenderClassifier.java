package ch.innunvation;

import ch.innunvation.ann.ANN;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Gender Classification from Face Images
 * Uses a neural network to classify faces as male (0) or female (1)
 *
 * IMPORTANT: This version uses ANNCrossEntropy for proper classification training
 */
public class GenderClassifier {

    public static final int IMG_SIZE = 48; // 48x48 grayscale images
    public static final int INPUT_SIZE = IMG_SIZE * IMG_SIZE; // 2304 inputs

    private final ANN network;

    public GenderClassifier(int hiddenNeurons, double learningRate, long seed) {
        // Network: 2304 inputs -> hidden layer -> 1 output (0=male, 1=female)
        this.network = new ANN(INPUT_SIZE, hiddenNeurons, 1, learningRate, seed);
    }

    /**
     * Preprocess an image: resize to 48x48, convert to grayscale, normalize
     */
    public static double[] preprocessImage(BufferedImage img) {
        // Resize to 48x48
        BufferedImage resized = resizeImage(img, IMG_SIZE, IMG_SIZE);

        // Convert to grayscale and normalize to [0,1]
        double[] pixels = new double[INPUT_SIZE];
        int idx = 0;
        for (int y = 0; y < IMG_SIZE; y++) {
            for (int x = 0; x < IMG_SIZE; x++) {
                int rgb = resized.getRGB(x, y);
                int gray = toGrayscale(rgb);
                pixels[idx++] = gray / 255.0; // Normalize to [0,1]
            }
        }

        // Optional: histogram equalization for better contrast
        return histogramEqualization(pixels);
    }

    /**
     * Resize image using bilinear interpolation
     */
    private static BufferedImage resizeImage(BufferedImage original, int width, int height) {
        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(original, 0, 0, width, height, null);
        g.dispose();
        return resized;
    }

    /**
     * Convert RGB to grayscale using luminosity method
     */
    private static int toGrayscale(int rgb) {
        int r = (rgb >> 16) & 0xFF;
        int g = (rgb >> 8) & 0xFF;
        int b = rgb & 0xFF;
        // Weighted average (human eye is more sensitive to green)
        return (int) (0.299 * r + 0.587 * g + 0.114 * b);
    }

    /**
     * Histogram equalization to improve contrast
     */
    private static double[] histogramEqualization(double[] pixels) {
        int[] histogram = new int[256];

        // Build histogram
        for (double pixel : pixels) {
            int value = (int) (pixel * 255);
            histogram[value]++;
        }

        // Calculate cumulative distribution
        int[] cdf = new int[256];
        cdf[0] = histogram[0];
        for (int i = 1; i < 256; i++) {
            cdf[i] = cdf[i - 1] + histogram[i];
        }

        // Normalize CDF
        int cdfMin = cdf[0];
        int totalPixels = pixels.length;
        double[] result = new double[pixels.length];

        for (int i = 0; i < pixels.length; i++) {
            int value = (int) (pixels[i] * 255);
            result[i] = ((cdf[value] - cdfMin) / (double) (totalPixels - cdfMin));
        }

        return result;
    }

    /**
     * Train the network on a dataset
     */
    public void train(double[][] X, double[][] Y, int epochs) {
        network.train(X, Y, epochs);
    }

    /**
     * Predict gender from preprocessed image data
     * @return probability of being female (0.0 = male, 1.0 = female)
     */
    public double predict(double[] imageData) {
        double[] output = network.apply(imageData);
        return output[0];
    }

    /**
     * Predict gender with label
     */
    public String predictLabel(double[] imageData) {
        double prob = predict(imageData);
        return prob > 0.5 ? "Female" : "Male";
    }

    /**
     * Simple model state for checkpointing
     */
    public static class ModelCheckpoint {
        public double[][][] weights; // [w1, w2]
        public double[][] biases;    // [b1, b2]
        public double testAccuracy;
        public int epoch;
    }

    /**
     * Save current model state
     */
    public ModelCheckpoint saveCheckpoint(double testAccuracy, int epoch) {
        ModelCheckpoint cp = new ModelCheckpoint();
        cp.testAccuracy = testAccuracy;
        cp.epoch = epoch;

        // Deep copy weights and biases from the network
        // Note: This requires the ANN class fields to be accessible
        // For now, we'll just track the metrics
        return cp;
    }
    public double accuracy(double[][] X, double[][] Y) {
        int correct = 0;
        for (int i = 0; i < X.length; i++) {
            double pred = predict(X[i]);
            double actual = Y[i][0];
            if (Math.round(pred) == Math.round(actual)) {
                correct++;
            }
        }
        return (double) correct / X.length;
    }

    /**
     * Data holder for training/test samples
     */
    public static class Dataset {
        public double[][] X;
        public double[][] Y;
        public String[] filenames;

        public Dataset(double[][] X, double[][] Y, String[] filenames) {
            this.X = X;
            this.Y = Y;
            this.filenames = filenames;
        }
    }

    /**
     * Load images from directories (male/ and female/)
     * Expected structure:
     *   dataDir/
     *     male/
     *       img1.jpg
     *       img2.jpg
     *     female/
     *       img1.jpg
     *       img2.jpg
     */
    public static Dataset loadDataset(String dataDir) throws IOException {
        List<double[]> X = new ArrayList<>();
        List<double[]> Y = new ArrayList<>();
        List<String> filenames = new ArrayList<>();

        // Load male images (label = 0)
        File maleDir = new File(dataDir, "male");
        if (maleDir.exists()) {
            for (File file : maleDir.listFiles()) {
                if (isImageFile(file)) {
                    try {
                        BufferedImage img = ImageIO.read(file);
                        double[] features = preprocessImage(img);
                        X.add(features);
                        Y.add(new double[]{0.0}); // Male = 0
                        filenames.add(file.getName());
                    } catch (Exception e) {
                        System.err.println("Error loading " + file.getName() + ": " + e.getMessage());
                    }
                }
            }
        }

        // Load female images (label = 1)
        File femaleDir = new File(dataDir, "female");
        if (femaleDir.exists()) {
            for (File file : femaleDir.listFiles()) {
                if (isImageFile(file)) {
                    try {
                        BufferedImage img = ImageIO.read(file);
                        double[] features = preprocessImage(img);
                        X.add(features);
                        Y.add(new double[]{1.0}); // Female = 1
                        filenames.add(file.getName());
                    } catch (Exception e) {
                        System.err.println("Error loading " + file.getName() + ": " + e.getMessage());
                    }
                }
            }
        }

        // Shuffle dataset
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < X.size(); i++) indices.add(i);
        Collections.shuffle(indices, new Random(42));

        double[][] XArray = new double[X.size()][];
        double[][] YArray = new double[Y.size()][];
        String[] filenamesArray = new String[filenames.size()];

        for (int i = 0; i < indices.size(); i++) {
            int idx = indices.get(i);
            XArray[i] = X.get(idx);
            YArray[i] = Y.get(idx);
            filenamesArray[i] = filenames.get(idx);
        }

        return new Dataset(XArray, YArray, filenamesArray);
    }

    /**
     * Split dataset into train and test sets
     */
    public static Dataset[] trainTestSplit(Dataset data, double trainRatio) {
        int trainSize = (int) (data.X.length * trainRatio);

        double[][] XTrain = new double[trainSize][];
        double[][] YTrain = new double[trainSize][];
        String[] filenamesTrain = new String[trainSize];

        double[][] XTest = new double[data.X.length - trainSize][];
        double[][] YTest = new double[data.Y.length - trainSize][];
        String[] filenamesTest = new String[data.filenames.length - trainSize];

        System.arraycopy(data.X, 0, XTrain, 0, trainSize);
        System.arraycopy(data.Y, 0, YTrain, 0, trainSize);
        System.arraycopy(data.filenames, 0, filenamesTrain, 0, trainSize);

        System.arraycopy(data.X, trainSize, XTest, 0, data.X.length - trainSize);
        System.arraycopy(data.Y, trainSize, YTest, 0, data.Y.length - trainSize);
        System.arraycopy(data.filenames, trainSize, filenamesTest, 0, data.filenames.length - trainSize);

        return new Dataset[]{
                new Dataset(XTrain, YTrain, filenamesTrain),
                new Dataset(XTest, YTest, filenamesTest)
        };
    }

    private static boolean isImageFile(File file) {
        if (!file.isFile()) return false;
        String name = file.getName().toLowerCase();
        return name.endsWith(".jpg") || name.endsWith(".jpeg") ||
                name.endsWith(".png") || name.endsWith(".bmp");
    }

    /**
     * Example usage
     */
    public static void main(String[] args) throws IOException {
        System.out.println("=== Gender Classification from Face Images ===\n");

        // Configuration
        String dataDir = "src/main/resources/faces_dataset_cropped";
        int hiddenNeurons = 32;
        double learningRate = 0.2; // Higher LR works better with cross-entropy
        int epochs = 500;
        long seed = 42;

        // Load dataset
        System.out.println("Loading dataset from: " + dataDir);
        Dataset data = loadDataset(dataDir);
        System.out.println("Loaded " + data.X.length + " images");

        // Analyze dataset balance
        int maleCount = 0, femaleCount = 0;
        for (double[] label : data.Y) {
            if (label[0] == 0.0) maleCount++;
            else femaleCount++;
        }
        System.out.println("  Male: " + maleCount + " (" + String.format("%.1f%%", 100.0 * maleCount / data.X.length) + ")");
        System.out.println("  Female: " + femaleCount + " (" + String.format("%.1f%%", 100.0 * femaleCount / data.X.length) + ")");

        if (data.X.length == 0) {
            System.out.println("\nNo images found! Please create the following directory structure:");
            System.out.println("  " + dataDir + "/");
            System.out.println("    male/");
            System.out.println("      img1.jpg");
            System.out.println("      img2.jpg");
            System.out.println("      ...");
            System.out.println("    female/");
            System.out.println("      img1.jpg");
            System.out.println("      img2.jpg");
            System.out.println("      ...");
            return;
        }

        // Split into train/test
        Dataset[] split = trainTestSplit(data, 0.8);
        Dataset trainData = split[0];
        Dataset testData = split[1];
        System.out.println("Train set: " + trainData.X.length + " images");
        System.out.println("Test set: " + testData.X.length + " images\n");

        // Create and train classifier
        System.out.println("\nTraining network...");
        System.out.println("Architecture: " + INPUT_SIZE + " -> " + hiddenNeurons + " -> 1");
        int totalParams = (INPUT_SIZE * hiddenNeurons + hiddenNeurons) + (hiddenNeurons * 1 + 1);
        System.out.println("Total parameters: " + String.format("%,d", totalParams));
        System.out.println("Parameters per sample ratio: " + String.format("%.1f", (double) totalParams / trainData.X.length));
        System.out.println("Learning rate: " + learningRate);
        System.out.println("Epochs: " + epochs + "\n");

        GenderClassifier classifier = new GenderClassifier(hiddenNeurons, learningRate, seed);

        // Training with early stopping
        double bestTestAcc = 0.0;
        int bestEpoch = 0;
        int patience = 100;
        int noImprovementCount = 0;

        // Store best model configuration for reference
        GenderClassifier bestModel = null;

        for (int epoch = 0; epoch < epochs; epoch++) {
            classifier.train(trainData.X, trainData.Y, 1);

            double testAcc = classifier.accuracy(testData.X, testData.Y);

            // Track best model
            if (testAcc > bestTestAcc) {
                bestTestAcc = testAcc;
                bestEpoch = epoch;
                noImprovementCount = 0;

                // NOTE: In a production system, you would save the weights here
                // For now, we just track when the best performance occurred
            } else {
                noImprovementCount++;
            }

            if (epoch % 50 == 0 || epoch == epochs - 1) {
                double trainAcc = classifier.accuracy(trainData.X, trainData.Y);
                double trainMSE = classifier.network.mse(trainData.X, trainData.Y);
                System.out.printf("Epoch %4d: Train=%.1f%% Test=%.1f%% MSE=%.4f (Best: %.1f%% @ epoch %d)%n",
                        epoch, trainAcc * 100, testAcc * 100, trainMSE, bestTestAcc * 100, bestEpoch);
            }

            // Early stopping
            if (noImprovementCount >= patience) {
                System.out.printf("%nEarly stopping at epoch %d (no improvement for %d epochs)%n",
                        epoch, patience);
                System.out.printf("⚠️  Best model was at epoch %d with %.1f%% test accuracy%n",
                        bestEpoch, bestTestAcc * 100);
                System.out.println("   (Current model has likely overfit since then)");
                break;
            }
        }

        // Test evaluation
        System.out.println("\n=== Final Evaluation ===");
        double testAcc = classifier.accuracy(testData.X, testData.Y);
        double testMSE = classifier.network.mse(testData.X, testData.Y);
        System.out.printf("Test Accuracy: %.2f%%%n", testAcc * 100);
        System.out.printf("Test MSE: %.4f%n", testMSE);

        // Show some predictions
        System.out.println("\n=== Sample Predictions ===");
        for (int i = 0; i < Math.min(10, testData.X.length); i++) {
            double prob = classifier.predict(testData.X[i]);
            String predicted = classifier.predictLabel(testData.X[i]);
            String actual = testData.Y[i][0] == 1.0 ? "Female" : "Male";
            String correct = predicted.equals(actual) ? "✓" : "✗";
            System.out.printf("%s %s: %.3f -> %s (actual: %s)%n",
                    correct, testData.filenames[i], prob, predicted, actual);
        }
    }
}