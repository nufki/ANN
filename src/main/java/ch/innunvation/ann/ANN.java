package ch.innunvation.ann;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class ANN {
    private final int nIn;
    private final int nHidden;
    private final int nOut;


    // Default learning rate (optional convenience)
    private final double defaultLearningRate;

    // Weights and biases:
    private final double[][] w1; // Weights from inputs → hidden layer (in the docu these are: w13, w14, w23, w24)
    private final double[] b1; // Biases of hidden neurons (in the docu these are: bias of H3, H4)
    private final double[][] w2; // Weights from hidden → output layer (in the docu these are: w35, w45)
    private final double[] b2; // Biases of output neurons (in the docu this is O5)

    private final Random rnd;
    
    // Weight history tracking for visualization
    private static class WeightSnapshot {
        final double w1;
        final double w2;
        final int updateCount;
        
        WeightSnapshot(double w1, double w2, int updateCount) {
            this.w1 = w1;
            this.w2 = w2;
            this.updateCount = updateCount;
        }
    }
    
    private static class WeightKey {
        final int hiddenIdx1, inputIdx1, hiddenIdx2, inputIdx2;
        
        WeightKey(int hiddenIdx1, int inputIdx1, int hiddenIdx2, int inputIdx2) {
            this.hiddenIdx1 = hiddenIdx1;
            this.inputIdx1 = inputIdx1;
            this.hiddenIdx2 = hiddenIdx2;
            this.inputIdx2 = inputIdx2;
        }
        
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            WeightKey weightKey = (WeightKey) o;
            return hiddenIdx1 == weightKey.hiddenIdx1 &&
                   inputIdx1 == weightKey.inputIdx1 &&
                   hiddenIdx2 == weightKey.hiddenIdx2 &&
                   inputIdx2 == weightKey.inputIdx2;
        }
        
        @Override
        public int hashCode() {
            int result = hiddenIdx1;
            result = 31 * result + inputIdx1;
            result = 31 * result + hiddenIdx2;
            result = 31 * result + inputIdx2;
            return result;
        }
    }
    
    private Map<WeightKey, List<WeightSnapshot>> weightHistories;
    private int updateCounter;
    private int totalUpdates;

    /**
     * Basic constructor (random seed not fixed, default LR = 0.1).
     */
    public ANN(int inputs, int hiddenNeurons, int outputNeurons) {
        this(inputs, hiddenNeurons, outputNeurons, 0.1, new Random());
    }

    /**
     * Basic constructor with Random (default LR = 0.1).
     */
    public ANN(int inputs, int hiddenNeurons, int outputNeurons, Random random) {
        this(inputs, hiddenNeurons, outputNeurons, 0.1, random);
    }

    /**
     * Convenience constructor matching your earlier usage:
     * ANN ann = new ANN(2, 6, 3, 0.3, 42);
     *
     * @param defaultLearningRate used by train(X, Y, epochs)
     * @param seed                random seed for reproducible initialization
     */
    public ANN(int inputs, int hiddenNeurons, int outputNeurons, double defaultLearningRate, long seed) {
        this(inputs, hiddenNeurons, outputNeurons, defaultLearningRate, new Random(seed));
    }

    /**
     * Full constructor: specify default LR and Random.
     */
    public ANN(int inputs, int hiddenNeurons, int outputNeurons, double defaultLearningRate, Random random) {
        if (inputs <= 0 || hiddenNeurons <= 0 || outputNeurons <= 0) {
            throw new IllegalArgumentException("All layer sizes must be > 0.");
        }
        if (defaultLearningRate <= 0) {
            throw new IllegalArgumentException("defaultLearningRate must be > 0.");
        }
        if (random == null) {
            throw new IllegalArgumentException("random must not be null.");
        }

        this.nIn = inputs;
        this.nHidden = hiddenNeurons;
        this.nOut = outputNeurons;
        this.defaultLearningRate = defaultLearningRate;
        this.rnd = random;

        this.w1 = new double[nHidden][nIn];
        this.b1 = new double[nHidden];
        this.w2 = new double[nOut][nHidden];
        this.b2 = new double[nOut];

        initWeights();
    }

    // Xavier/Glorot-ish uniform init for sigmoid
    // Drawing from a gaussian distribution with a mean of 0
    // range = sqrt (6 / (n_inputs + n_outputs))
    private void initWeights() {
        initMatrix(w1, Math.sqrt(6.0 / (nIn + nHidden)));
        initMatrix(w2, Math.sqrt(6.0 / (nHidden + nOut)));

        // biases start at 0
        for (int i = 0; i < nHidden; i++) b1[i] = 0.0;
        for (int i = 0; i < nOut; i++) b2[i] = 0.0;
    }

    private void initMatrix(double[][] m, double limit) {
        for (int r = 0; r < m.length; r++) {
            for (int c = 0; c < m[r].length; c++) {
                m[r][c] = uniform(-limit, limit);
            }
        }
    }

    private double uniform(double a, double b) {
        return a + (b - a) * rnd.nextDouble();
    }

    /**
     * Forward pass: returns output activations.
     */
    public double[] apply(double[] x) {
        if (x == null || x.length != nIn) {
            throw new IllegalArgumentException("Input must have length " + nIn);
        }

        double[] h = new double[nHidden];
        for (int i = 0; i < nHidden; i++) {
            double z = b1[i];
            for (int j = 0; j < nIn; j++) z += w1[i][j] * x[j];
            h[i] = sigmoid(z);
        }

        double[] y = new double[nOut];
        for (int k = 0; k < nOut; k++) {
            double z = b2[k];
            for (int i = 0; i < nHidden; i++) z += w2[k][i] * h[i];
            y[k] = sigmoid(z);
        }
        return y;
    }

    /**
     * Trains with SGD on mean squared error (MSE) using backprop,
     * using the default learning rate provided in the constructor.
     */
    public void train(double[][] X, double[][] Y, int epochs) {
        train(X, Y, epochs, this.defaultLearningRate);
    }

    /**
     * Enables weight history tracking for a specific weight pair during training.
     * Can be called multiple times to track multiple weight pairs.
     * @param hiddenIdx1 hidden neuron index for first weight
     * @param inputIdx1 input index for first weight
     * @param hiddenIdx2 hidden neuron index for second weight
     * @param inputIdx2 input index for second weight
     */
    public void enableWeightHistoryTracking(int hiddenIdx1, int inputIdx1, int hiddenIdx2, int inputIdx2) {
        if (hiddenIdx1 < 0 || hiddenIdx1 >= nHidden || inputIdx1 < 0 || inputIdx1 >= nIn ||
            hiddenIdx2 < 0 || hiddenIdx2 >= nHidden || inputIdx2 < 0 || inputIdx2 >= nIn) {
            throw new IllegalArgumentException("Invalid weight indices for tracking");
        }
        if (weightHistories == null) {
            weightHistories = new HashMap<>();
        }
        WeightKey key = new WeightKey(hiddenIdx1, inputIdx1, hiddenIdx2, inputIdx2);
        weightHistories.put(key, new ArrayList<>());
    }
    
    /**
     * Disables weight history tracking for all weight pairs.
     */
    public void disableWeightHistoryTracking() {
        this.weightHistories = null;
    }
    
    /**
     * Gets the weight history for a specific weight pair if tracking was enabled during training.
     * @param hiddenIdx1 hidden neuron index for first weight
     * @param inputIdx1 input index for first weight
     * @param hiddenIdx2 hidden neuron index for second weight
     * @param inputIdx2 input index for second weight
     * @return list of [w1, w2] pairs, or null if tracking was not enabled for this pair
     */
    public List<double[]> getWeightHistory(int hiddenIdx1, int inputIdx1, int hiddenIdx2, int inputIdx2) {
        if (weightHistories == null) {
            return null;
        }
        WeightKey key = new WeightKey(hiddenIdx1, inputIdx1, hiddenIdx2, inputIdx2);
        List<WeightSnapshot> history = weightHistories.get(key);
        if (history == null) {
            return null;
        }
        List<double[]> result = new ArrayList<>();
        for (WeightSnapshot snap : history) {
            result.add(new double[]{snap.w1, snap.w2});
        }
        return result;
    }
    
    /**
     * Trains with SGD on mean squared error (MSE) using backprop.
     * X: samples x nIn
     * Y: samples x nOut
     */
    public void train(double[][] X, double[][] Y, int epochs, double learningRate) {
        if (X == null || Y == null) throw new IllegalArgumentException("X/Y must not be null.");
        if (X.length != Y.length) throw new IllegalArgumentException("X and Y must have same #samples.");
        if (epochs <= 0) throw new IllegalArgumentException("epochs must be > 0.");
        if (learningRate <= 0) throw new IllegalArgumentException("learningRate must be > 0.");

        // Initialize tracking if enabled
        if (weightHistories != null && !weightHistories.isEmpty()) {
            totalUpdates = epochs * X.length;
            updateCounter = 0;
            // Record initial weights for all tracked pairs
            for (Map.Entry<WeightKey, List<WeightSnapshot>> entry : weightHistories.entrySet()) {
                WeightKey key = entry.getKey();
                entry.getValue().add(new WeightSnapshot(
                    w1[key.hiddenIdx1][key.inputIdx1],
                    w1[key.hiddenIdx2][key.inputIdx2],
                    0
                ));
            }
        }

        for (int e = 0; e < epochs; e++) {
            for (int s = 0; s < X.length; s++) {
                double[] x = X[s]; // input X[s]
                double[] t = Y[s]; // Output Y[s]

                if (x == null || x.length != nIn) {
                    throw new IllegalArgumentException("X[" + s + "] length != " + nIn);
                }
                if (t == null || t.length != nOut) {
                    throw new IllegalArgumentException("Y[" + s + "] length != " + nOut);
                }

                // Forward propagation (store activations for backprop)
                double[] h = new double[nHidden];
                // Calculate front to hidden neuron activations
                for (int i = 0; i < nHidden; i++) {
                    double z = b1[i];
                    for (int j = 0; j < nIn; j++) z += w1[i][j] * x[j];
                    h[i] = sigmoid(z);
                }

                // Calculate hidden to output neuron activations
                double[] y = new double[nOut];
                for (int k = 0; k < nOut; k++) {
                    double z = b2[k];
                    for (int i = 0; i < nHidden; i++) z += w2[k][i] * h[i];
                    y[k] = sigmoid(z);
                }

                // Calculate backpropagation deltas (output neurons to hidden)
                // Notes from my end: The code here uses a different notion than the theory attached as it calculates
                // y[k] - t[k] instead of t[k] - y[k]. However, in the weight change routine, the weight delta is subtracted
                // rather than summed which makes it identical. in fact, the loss function is defined as:
                // L=1/2 * (y − t)^2 and the gradient thereof ⇒ ∂L/∂y = (y - t)
                double[] deltaOut = new double[nOut];
                for (int k = 0; k < nOut; k++) {
                    double error = (y[k] - t[k]);
                    double sigmoid_derivative = y[k] * (1.0 - y[k]);
                    // Clip sigmoid derivative to avoid vanishing gradients
                    // sigmoid_derivative = Math.max(sigmoid_derivative, 1e-7);
                    deltaOut[k] = sigmoid_derivative * error;
                    // Gradient clipping to prevent explosion
                    // deltaOut[k] = Math.max(-5.0, Math.min(5.0, deltaOut[k]));
                }

                // Calculate backpropagation deltas (hidden neurons to input)
                double[] deltaHidden = new double[nHidden];
                for (int i = 0; i < nHidden; i++) {
                    double sum = 0.0;
                    for (int k = 0; k < nOut; k++) {
                        sum += w2[k][i] * deltaOut[k];
                    }
                    double sigmoid_derivative = h[i] * (1.0 - h[i]);
                    // Clip sigmoid derivative to avoid vanishing gradients
                    // sigmoid_derivative = Math.max(sigmoid_derivative, 1e-7);
                    deltaHidden[i] = sum * sigmoid_derivative;

                    // Gradient clipping to prevent explosion
                    // deltaHidden[i] = Math.max(-5.0, Math.min(5.0, deltaHidden[i]));
                }

                // Calculate gradient step (new weights) - hidden to output
                for (int k = 0; k < nOut; k++) {
                    for (int i = 0; i < nHidden; i++) {
                        w2[k][i] -= learningRate * deltaOut[k] * h[i];
                    }
                    b2[k] -= learningRate * deltaOut[k];
                }

                // Calculate gradient step (new weights) - input to hidden
                for (int i = 0; i < nHidden; i++) {
                    for (int j = 0; j < nIn; j++) {
                        w1[i][j] -= learningRate * deltaHidden[i] * x[j];
                    }
                    b1[i] -= learningRate * deltaHidden[i];
                }
                
                // Asymptotic sampling of weights for visualization
                // Sample more frequently at the beginning, less frequently as training progresses
                if (weightHistories != null && !weightHistories.isEmpty()) {
                    updateCounter++;
                    // Asymptotic sampling: interval grows exponentially
                    // Early: sample every ~10 updates, later: sample every ~1000+ updates
                    double progress = (double) updateCounter / totalUpdates;
                    // Base interval grows from 1 to maxInterval based on progress
                    // Using exponential growth: baseInterval = 1 + (maxInterval - 1) * (1 - exp(-progress * k))
                    double maxInterval = 1000.0; // Maximum sampling interval near the end
                    double k = 3.0; // Controls growth rate
                    double baseInterval = 1.0 + (maxInterval - 1.0) * (1.0 - Math.exp(-progress * k));
                    int currentInterval = (int) Math.max(1, Math.round(baseInterval));
                    
                    // Sample at regular intervals, but interval increases over time
                    if (updateCounter % currentInterval == 0 || updateCounter == totalUpdates) {
                        // Record for all tracked weight pairs
                        for (Map.Entry<WeightKey, List<WeightSnapshot>> entry : weightHistories.entrySet()) {
                            WeightKey key = entry.getKey();
                            entry.getValue().add(new WeightSnapshot(
                                w1[key.hiddenIdx1][key.inputIdx1],
                                w1[key.hiddenIdx2][key.inputIdx2],
                                updateCounter
                            ));
                        }
                    }
                }
            }
        }
    }

    private static double sigmoid(double z) {
        // avoid overflow a bit
        if (z >= 0) {
            double ez = Math.exp(-z);
            return 1.0 / (1.0 + ez);
        } else {
            double ez = Math.exp(z);
            return ez / (1.0 + ez);
        }
    }

    // --- optional helper: compute average MSE for monitoring ---
    public double mse(double[][] X, double[][] Y) {
        double sum = 0.0;
        int n = X.length;
        for (int s = 0; s < n; s++) {
            double[] y = apply(X[s]);
            for (int k = 0; k < nOut; k++) {
                double d = y[k] - Y[s][k];
                sum += d * d;
            }
        }
        return sum / (n * nOut);
    }
    
    /**
     * Computes MSE with temporarily modified weights for error surface visualization.
     * This method temporarily modifies specific weights, computes error, then restores them.
     * NOTE: This modifies the network weights temporarily - use with caution in multi-threaded environments.
     * 
     * @param X training inputs
     * @param Y training outputs
     * @param hiddenIdx which hidden neuron's weights to modify (0-based)
     * @param inputIdx1 first input weight index to modify
     * @param inputIdx2 second input weight index to modify
     * @param w1Value new value for first weight
     * @param w2Value new value for second weight
     * @return MSE with modified weights
     */
    public synchronized double mseWithModifiedWeights(double[][] X, double[][] Y, 
                                         int hiddenIdx, int inputIdx1, int inputIdx2,
                                         double w1Value, double w2Value) {
        // Save original weights
        double originalW1 = w1[hiddenIdx][inputIdx1];
        double originalW2 = w1[hiddenIdx][inputIdx2];
        
        try {
            // Temporarily modify weights
            w1[hiddenIdx][inputIdx1] = w1Value;
            w1[hiddenIdx][inputIdx2] = w2Value;
            
            // Compute error
            return mse(X, Y);
        } finally {
            // Always restore original weights, even if an exception occurs
            w1[hiddenIdx][inputIdx1] = originalW1;
            w1[hiddenIdx][inputIdx2] = originalW2;
        }
    }
    
    /**
     * Gets a specific weight value (for visualization purposes)
     */
    public double getWeight(int layer, int fromIdx, int toIdx) {
        if (layer == 1) {
            return w1[toIdx][fromIdx];
        } else if (layer == 2) {
            return w2[toIdx][fromIdx];
        }
        throw new IllegalArgumentException("Invalid layer: " + layer);
    }
    
    /**
     * Computes MSE with two weights from potentially different hidden neurons modified.
     * This is a more general version for error surface visualization.
     */
    public synchronized double mseWithTwoModifiedWeights(double[][] X, double[][] Y,
                                                        int hiddenIdx1, int inputIdx1, double w1Value,
                                                        int hiddenIdx2, int inputIdx2, double w2Value) {
        // Save original weights
        double originalW1 = w1[hiddenIdx1][inputIdx1];
        double originalW2 = w1[hiddenIdx2][inputIdx2];
        
        try {
            // Temporarily modify weights
            w1[hiddenIdx1][inputIdx1] = w1Value;
            w1[hiddenIdx2][inputIdx2] = w2Value;
            
            // Compute error
            return mse(X, Y);
        } finally {
            // Always restore original weights, even if an exception occurs
            w1[hiddenIdx1][inputIdx1] = originalW1;
            w1[hiddenIdx2][inputIdx2] = originalW2;
        }
    }
}


