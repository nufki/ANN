package ch.innunvation.ann;

import java.util.Random;

/**
 * ANN with Binary Cross-Entropy Loss - MUCH better for classification!
 * Loss = -[t*log(y) + (1-t)*log(1-y)]
 */
public class ANNCrossEntropy {
    private final int nIn;
    private final int nHidden;
    private final int nOut;
    private final double defaultLearningRate;

    private final double[][] w1;
    private final double[] b1;
    private final double[][] w2;
    private final double[] b2;

    private final Random rnd;

    public ANNCrossEntropy(int inputs, int hiddenNeurons, int outputNeurons, double defaultLearningRate, long seed) {
        if (inputs <= 0 || hiddenNeurons <= 0 || outputNeurons <= 0) {
            throw new IllegalArgumentException("All layer sizes must be > 0.");
        }
        if (defaultLearningRate <= 0) {
            throw new IllegalArgumentException("defaultLearningRate must be > 0.");
        }

        this.nIn = inputs;
        this.nHidden = hiddenNeurons;
        this.nOut = outputNeurons;
        this.defaultLearningRate = defaultLearningRate;
        this.rnd = new Random(seed);

        this.w1 = new double[nHidden][nIn];
        this.b1 = new double[nHidden];
        this.w2 = new double[nOut][nHidden];
        this.b2 = new double[nOut];

        initWeights();
    }

    private void initWeights() {
        // Xavier/Glorot initialization
        double limitW1 = Math.sqrt(6.0 / (nIn + nHidden));
        double limitW2 = Math.sqrt(6.0 / (nHidden + nOut));

        initMatrix(w1, limitW1);
        initMatrix(w2, limitW2);

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

    public void train(double[][] X, double[][] Y, int epochs) {
        train(X, Y, epochs, this.defaultLearningRate);
    }

    /**
     * Training with Binary Cross-Entropy Loss
     * Gradient simplifies beautifully: δ_out = y - t (no sigmoid derivative needed!)
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

                // Forward pass
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

                // CROSS-ENTROPY: Output gradient simplifies to just (y - t)!
                // This is the magic: the sigmoid derivative cancels out
                double[] deltaOut = new double[nOut];
                for (int k = 0; k < nOut; k++) {
                    deltaOut[k] = y[k] - t[k];
                }

                // Hidden layer gradients (unchanged)
                double[] deltaHidden = new double[nHidden];
                for (int i = 0; i < nHidden; i++) {
                    double sum = 0.0;
                    for (int k = 0; k < nOut; k++) {
                        sum += w2[k][i] * deltaOut[k];
                    }
                    deltaHidden[i] = sum * h[i] * (1.0 - h[i]);
                }

                // Update weights
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
        if (z >= 0) {
            double ez = Math.exp(-z);
            return 1.0 / (1.0 + ez);
        } else {
            double ez = Math.exp(z);
            return ez / (1.0 + ez);
        }
    }

    /**
     * Binary Cross-Entropy Loss (proper metric for classification)
     */
    public double crossEntropyLoss(double[][] X, double[][] Y) {
        double sum = 0.0;
        int n = X.length;
        final double epsilon = 1e-7; // Prevent log(0)

        for (int s = 0; s < n; s++) {
            double[] y = apply(X[s]);
            for (int k = 0; k < nOut; k++) {
                double t = Y[s][k];
                double pred = Math.max(epsilon, Math.min(1.0 - epsilon, y[k]));
                sum -= t * Math.log(pred) + (1.0 - t) * Math.log(1.0 - pred);
            }
        }
        return sum / n;
    }

    /**
     * Still provide MSE for compatibility
     */
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