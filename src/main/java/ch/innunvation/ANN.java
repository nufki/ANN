package ch.innunvation;

import java.util.Random;

public class ANN {
    private final int nIn, nHidden, nOut;

    // Default learning rate (optional convenience)
    private final double defaultLearningRate;

    // Weights and biases:
    // w1: hidden x input
    // b1: hidden
    // w2: output x hidden
    // b2: output
    private final double[][] w1;
    private final double[] b1;
    private final double[][] w2;
    private final double[] b2;

    private final Random rnd;

    // --- Constructors ---

    /** Basic constructor (random seed not fixed, default LR = 0.1). */
    public ANN(int inputs, int hiddenNeurons, int outputNeurons) {
        this(inputs, hiddenNeurons, outputNeurons, 0.1, new Random());
    }

    /** Basic constructor with Random (default LR = 0.1). */
    public ANN(int inputs, int hiddenNeurons, int outputNeurons, Random random) {
        this(inputs, hiddenNeurons, outputNeurons, 0.1, random);
    }

    /**
     * Convenience constructor matching your earlier usage:
     * ANN ann = new ANN(2, 6, 3, 0.3, 42);
     *
     * @param defaultLearningRate used by train(X, Y, epochs)
     * @param seed random seed for reproducible initialization
     */
    public ANN(int inputs, int hiddenNeurons, int outputNeurons, double defaultLearningRate, long seed) {
        this(inputs, hiddenNeurons, outputNeurons, defaultLearningRate, new Random(seed));
    }

    /** Full constructor: specify default LR and Random. */
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

    /** Forward pass: returns output activations. */
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
     * Trains with SGD on mean squared error (MSE) using backprop.
     * X: samples x nIn
     * Y: samples x nOut
     */
    public void train(double[][] X, double[][] Y, int epochs, double learningRate) {
        if (X == null || Y == null) throw new IllegalArgumentException("X/Y must not be null.");
        if (X.length != Y.length) throw new IllegalArgumentException("X and Y must have same #samples.");
        if (epochs <= 0) throw new IllegalArgumentException("epochs must be > 0.");
        if (learningRate <= 0) throw new IllegalArgumentException("learningRate must be > 0.");

        for (int e = 0; e < epochs; e++) {
            for (int s = 0; s < X.length; s++) {
                double[] x = X[s];
                double[] t = Y[s];

                if (x == null || x.length != nIn) {
                    throw new IllegalArgumentException("X[" + s + "] length != " + nIn);
                }
                if (t == null || t.length != nOut) {
                    throw new IllegalArgumentException("Y[" + s + "] length != " + nOut);
                }

                // ----- forward (store activations for backprop) -----
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

                // ----- backprop deltas -----
                double[] deltaOut = new double[nOut];
                for (int k = 0; k < nOut; k++) {
                    double dL_dy = (y[k] - t[k]);
                    deltaOut[k] = dL_dy * sigmoidPrimeFromActivation(y[k]);
                }

                double[] deltaHidden = new double[nHidden];
                for (int i = 0; i < nHidden; i++) {
                    double sum = 0.0;
                    for (int k = 0; k < nOut; k++) {
                        sum += w2[k][i] * deltaOut[k];
                    }
                    deltaHidden[i] = sum * sigmoidPrimeFromActivation(h[i]);
                }

                // ----- gradient step -----
                for (int k = 0; k < nOut; k++) {
                    for (int i = 0; i < nHidden; i++) {
                        w2[k][i] -= learningRate * deltaOut[k] * h[i];
                    }
                    b2[k] -= learningRate * deltaOut[k];
                }

                for (int i = 0; i < nHidden; i++) {
                    for (int j = 0; j < nIn; j++) {
                        w1[i][j] -= learningRate * deltaHidden[i] * x[j];
                    }
                    b1[i] -= learningRate * deltaHidden[i];
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

    private static double sigmoidPrimeFromActivation(double a) {
        return a * (1.0 - a);
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
}


