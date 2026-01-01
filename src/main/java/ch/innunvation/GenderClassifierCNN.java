package ch.innunvation;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

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
 * Gender Classification using Convolutional Neural Network (CNN)
 * Powered by Deeplearning4j
 *
 * Architecture:
 * Input: 48x48x1 grayscale images
 * Conv1: 16 filters, 3x3, ReLU
 * Pool1: 2x2 max pooling
 * Conv2: 32 filters, 3x3, ReLU
 * Pool2: 2x2 max pooling
 * Dense: 64 neurons, ReLU
 * Output: 1 neuron, Sigmoid (binary classification)
 */
public class GenderClassifierCNN {

    public static final int IMG_SIZE = 48;
    public static final int CHANNELS = 1; // Grayscale
    private static final int SEED = 42;

    private MultiLayerNetwork model;

    public GenderClassifierCNN() {
        buildModel();
    }

    /**
     * Build the CNN architecture
     */
    private void buildModel() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(SEED)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001)) // Learning rate
                .list()
                // Layer 0: Convolutional layer - 16 filters, 3x3 kernel
                .layer(0, new ConvolutionLayer.Builder(3, 3)
                        .nIn(CHANNELS)
                        .nOut(16)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                // Layer 1: Max pooling - 2x2
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                // Layer 2: Convolutional layer - 32 filters, 3x3 kernel
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .nOut(32)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                // Layer 3: Max pooling - 2x2
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                // Layer 4: Fully connected layer - 64 neurons
                .layer(4, new DenseLayer.Builder()
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                // Layer 5: Output layer - 1 neuron (binary classification)
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .nOut(1)
                        .activation(Activation.SIGMOID)
                        .build())
                .setInputType(InputType.convolutionalFlat(IMG_SIZE, IMG_SIZE, CHANNELS))
                .build();

        model = new MultiLayerNetwork(conf);
        model.init();

        // Print every 10 iterations
        model.setListeners(new ScoreIterationListener(10));

        System.out.println("CNN Architecture:");
        System.out.println(model.summary());
    }

    /**
     * Train the model with mini-batch gradient descent
     */
    public void train(INDArray features, INDArray labels, int epochs, int batchSize) {
        int numSamples = (int) features.rows();

        System.out.println("\nTraining CNN...");
        System.out.println("Epochs: " + epochs);
        System.out.println("Batch size: " + batchSize);
        System.out.println("Training samples: " + numSamples);
        System.out.println();

        for (int epoch = 0; epoch < epochs; epoch++) {
            // Shuffle indices for each epoch
            int[] indices = new int[numSamples];
            for (int i = 0; i < numSamples; i++) indices[i] = i;
            shuffleArray(indices);

            // Train on mini-batches
            for (int i = 0; i < numSamples; i += batchSize) {
                int end = Math.min(i + batchSize, numSamples);
                int actualBatchSize = end - i;

                // Get batch
                INDArray batchFeatures = Nd4j.create(actualBatchSize, features.columns());
                INDArray batchLabels = Nd4j.create(actualBatchSize, 1);

                for (int j = 0; j < actualBatchSize; j++) {
                    int idx = indices[i + j];
                    batchFeatures.putRow(j, features.getRow(idx));
                    batchLabels.putRow(j, labels.getRow(idx));
                }

                DataSet batch = new DataSet(batchFeatures, batchLabels);
                /**
                 * Forward pass - Computes predictions for all samples in the batch
                 * Loss calculation - Compares predictions to actual labels
                 * Backward pass - Computes gradients via backpropagation
                 * Weight update - Updates all weights using the optimizer (Adam in our case)
                 */
                model.fit(batch);
            }

            if (epoch % 5 == 0 || epoch == epochs - 1) {
                double score = model.score();
                System.out.printf("Epoch %3d: Loss = %.4f%n", epoch, score);
            }
        }
    }

    private void shuffleArray(int[] array) {
        Random rnd = new Random(SEED);
        for (int i = array.length - 1; i > 0; i--) {
            int index = rnd.nextInt(i + 1);
            int temp = array[index];
            array[index] = array[i];
            array[i] = temp;
        }
    }

    /**
     * Predict gender for a single image
     */
    public double predict(INDArray features) {
        // Ensure features are in the right shape [1, 2304]
        if (features.rank() == 1) {
            features = features.reshape(1, features.length());
        }
        INDArray output = model.output(features);
        return output.getDouble(0);
    }

    /**
     * Calculate accuracy on a dataset
     */
    public double accuracy(INDArray features, INDArray labels) {
        INDArray predictions = model.output(features);
        int correct = 0;

        for (int i = 0; i < predictions.rows(); i++) {
            double pred = predictions.getDouble(i) > 0.5 ? 1.0 : 0.0;
            double actual = labels.getDouble(i);
            if (pred == actual) {
                correct++;
            }
        }

        return (double) correct / predictions.rows();
    }

    /**
     * Preprocess image to grayscale array
     */
    public static double[] preprocessImage(BufferedImage img) {
        // Resize to IMG_SIZE x IMG_SIZE
        BufferedImage resized = new BufferedImage(IMG_SIZE, IMG_SIZE, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = resized.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(img, 0, 0, IMG_SIZE, IMG_SIZE, null);
        g.dispose();

        // Convert to normalized array
        double[] pixels = new double[IMG_SIZE * IMG_SIZE];
        int idx = 0;
        for (int y = 0; y < IMG_SIZE; y++) {
            for (int x = 0; x < IMG_SIZE; x++) {
                int rgb = resized.getRGB(x, y);
                int gray = (rgb >> 16) & 0xFF; // Get red channel (all same in grayscale)
                pixels[idx++] = gray / 255.0; // Normalize to [0,1]
            }
        }

        return pixels;
    }

    /**
     * Load dataset from directories
     */
    public static class Dataset {
        public INDArray features;
        public INDArray labels;
        public String[] filenames; // Added filenames
        public int numSamples;

        public Dataset(INDArray features, INDArray labels, String[] filenames) {
            this.features = features;
            this.labels = labels;
            this.filenames = filenames;
            this.numSamples = (int) features.rows();
        }
    }

    public static Dataset loadDataset(String dataDir) throws IOException {
        List<double[]> featuresList = new ArrayList<>();
        List<Double> labelsList = new ArrayList<>();
        List<String> filenamesList = new ArrayList<>();

        // Load male images (label = 0)
        File maleDir = new File(dataDir, "male");
        if (maleDir.exists()) {
            File[] files = maleDir.listFiles((d, n) ->
                    n.toLowerCase().endsWith(".jpg") || n.toLowerCase().endsWith(".png"));
            if (files != null) {
                for (File file : files) {
                    try {
                        BufferedImage img = ImageIO.read(file);
                        double[] features = preprocessImage(img);
                        featuresList.add(features);
                        labelsList.add(0.0); // Male
                        filenamesList.add(file.getName());
                    } catch (Exception e) {
                        System.err.println("Error loading " + file.getName() + ": " + e.getMessage());
                    }
                }
            }
        }

        // Load female images (label = 1)
        File femaleDir = new File(dataDir, "female");
        if (femaleDir.exists()) {
            File[] files = femaleDir.listFiles((d, n) ->
                    n.toLowerCase().endsWith(".jpg") || n.toLowerCase().endsWith(".png"));
            if (files != null) {
                for (File file : files) {
                    try {
                        BufferedImage img = ImageIO.read(file);
                        double[] features = preprocessImage(img);
                        featuresList.add(features);
                        labelsList.add(1.0); // Female
                        filenamesList.add(file.getName());
                    } catch (Exception e) {
                        System.err.println("Error loading " + file.getName() + ": " + e.getMessage());
                    }
                }
            }
        }

        // Shuffle
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < featuresList.size(); i++) indices.add(i);
        Collections.shuffle(indices, new Random(SEED));

        // Convert to INDArray
        int numSamples = featuresList.size();
        double[][] featuresArray = new double[numSamples][];
        double[][] labelsArray = new double[numSamples][1];
        String[] filenamesArray = new String[numSamples];

        for (int i = 0; i < numSamples; i++) {
            int idx = indices.get(i);
            featuresArray[i] = featuresList.get(idx);
            labelsArray[i][0] = labelsList.get(idx);
            filenamesArray[i] = filenamesList.get(idx);
        }

        INDArray features = Nd4j.create(featuresArray);
        INDArray labels = Nd4j.create(labelsArray);

        return new Dataset(features, labels, filenamesArray);
    }

    /**
     * Split dataset into train and test
     */
    public static Dataset[] trainTestSplit(Dataset data, double trainRatio) {
        int trainSize = (int) (data.numSamples * trainRatio);

        // Use NDArrayIndex for proper slicing
        INDArray trainFeatures = data.features.get(
                NDArrayIndex.interval(0, trainSize),
                NDArrayIndex.all()
        );
        INDArray trainLabels = data.labels.get(
                NDArrayIndex.interval(0, trainSize),
                NDArrayIndex.all()
        );
        String[] trainFilenames = new String[trainSize];
        System.arraycopy(data.filenames, 0, trainFilenames, 0, trainSize);

        INDArray testFeatures = data.features.get(
                NDArrayIndex.interval(trainSize, data.numSamples),
                NDArrayIndex.all()
        );
        INDArray testLabels = data.labels.get(
                NDArrayIndex.interval(trainSize, data.numSamples),
                NDArrayIndex.all()
        );
        String[] testFilenames = new String[data.numSamples - trainSize];
        System.arraycopy(data.filenames, trainSize, testFilenames, 0, data.numSamples - trainSize);

        return new Dataset[]{
                new Dataset(trainFeatures, trainLabels, trainFilenames),
                new Dataset(testFeatures, testLabels, testFilenames)
        };
    }

    public static void main(String[] args) throws IOException {
        System.out.println("=== Gender Classification with CNN (Deeplearning4j) ===\n");

        // Configuration
        String dataDir = "src/main/resources/faces_dataset_cropped";
        int epochs = 100; // Increased from 50
        int batchSize = 32; // Can use larger batch now

        // Load dataset
        System.out.println("Loading dataset from: " + dataDir);
        Dataset data = loadDataset(dataDir);
        System.out.println("Loaded " + data.numSamples + " images\n");

        if (data.numSamples == 0) {
            System.out.println("No images found! Make sure the dataset exists.");
            return;
        }

        // Split into train/test
        Dataset[] split = trainTestSplit(data, 0.8);
        Dataset trainData = split[0];
        Dataset testData = split[1];

        System.out.println("Train set: " + trainData.numSamples + " images");
        System.out.println("Test set: " + testData.numSamples + " images\n");

        // Create and train model
        GenderClassifierCNN classifier = new GenderClassifierCNN();

        // Train
        long startTime = System.currentTimeMillis();
        classifier.train(trainData.features, trainData.labels, epochs, batchSize);
        long trainingTime = (System.currentTimeMillis() - startTime) / 1000;

        // Evaluate
        System.out.println("\n=== Evaluation ===");
        double trainAcc = classifier.accuracy(trainData.features, trainData.labels);
        double testAcc = classifier.accuracy(testData.features, testData.labels);

        System.out.printf("Training accuracy: %.2f%%%n", trainAcc * 100);
        System.out.printf("Test accuracy: %.2f%%%n", testAcc * 100);
        System.out.printf("Training time: %d seconds%n", trainingTime);

        // Show some predictions
        System.out.println("\n=== Sample Predictions (showing 10 out of " + testData.numSamples + " test samples) ===");
        for (int i = 0; i < Math.min(10, testData.numSamples); i++) {
            INDArray sample = testData.features.getRow(i);
            double prob = classifier.predict(sample);
            String predicted = prob > 0.5 ? "Female" : "Male";
            String actual = testData.labels.getDouble(i) == 1.0 ? "Female" : "Male";
            String correct = predicted.equals(actual) ? "✓" : "✗";
            System.out.printf("%s %s: %.3f -> %s (actual: %s)%n",
                    correct, testData.filenames[i], prob, predicted, actual);
        }
    }
}