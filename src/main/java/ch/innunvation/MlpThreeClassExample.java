package ch.innunvation;

import java.util.Arrays;


public class MlpThreeClassExample {

    public static void main(String[] args) {
        // 2 inputs, 6 hidden neurons, 3 outputs, learningRate=0.3, seed=42
        ANN ann = new ANN(2, 6, 3, 0.3, 42);

        // Training data: N samples, each with 2 inputs
        // Labels: N samples, each with 3 outputs (one-hot)
        double[][] X = {
                // Class 0 region (near 0,0)
                {0.05, 0.05},
                {0.10, 0.00},
                {0.00, 0.15},
                {0.12, 0.08},
                {0.20, 0.10},

                // Class 1 region (near 1,0)
                {0.90, 0.05},
                {1.00, 0.10},
                {0.85, 0.00},
                {0.95, 0.15},
                {0.80, 0.10},

                // Class 2 region (near 0,1)
                {0.05, 0.90},
                {0.10, 1.00},
                {0.00, 0.85},
                {0.15, 0.95},
                {0.10, 0.80},
        };

        double[][] Y = {
                // Class 0 -> [1,0,0]
                {1, 0, 0},
                {1, 0, 0},
                {1, 0, 0},
                {1, 0, 0},
                {1, 0, 0},

                // Class 1 -> [0,1,0]
                {0, 1, 0},
                {0, 1, 0},
                {0, 1, 0},
                {0, 1, 0},
                {0, 1, 0},

                // Class 2 -> [0,0,1]
                {0, 0, 1},
                {0, 0, 1},
                {0, 0, 1},
                {0, 0, 1},
                {0, 0, 1},
        };

        // Train
        int epochs = 3000;
        ann.train(X, Y, epochs, 1);

        // Test a few points
        test(ann, new double[]{0.08, 0.10}); // should be class 0-ish
        test(ann, new double[]{0.92, 0.08}); // should be class 1-ish
        test(ann, new double[]{0.05, 0.95}); // should be class 2-ish

        // Some "in-between" points
        test(ann, new double[]{0.60, 0.10}); // likely class 1 (closer to (1,0))
        test(ann, new double[]{0.10, 0.60}); // likely class 2 (closer to (0,1))
        test(ann, new double[]{0.35, 0.35}); // depends on your learned boundaries
    }

    private static void test(ANN ann, double[] x) {
        double[] out = ann.apply(x); // expects length 3
        int predicted = argMax(out);

        System.out.println("x=" + Arrays.toString(x)
                + " -> out=" + Arrays.toString(round3(out))
                + " predictedClass=" + predicted);
    }

    private static int argMax(double[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[best]) best = i;
        }
        return best;
    }

    private static double[] round3(double[] v) {
        double[] r = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            r[i] = Math.round(v[i] * 1000.0) / 1000.0;
        }
        return r;
    }
}
