package ch.innunvation;

import ch.innunvation.ann.ANN;
import ch.innunvation.ui.BoundaryPanelMulti;

import javax.swing.*;
import java.util.Arrays;

public class ThreeClassExample2D {

    public static void main(String[] args) {
        // If your ANN has (inputs, hidden, outputs, learningRate, seed) ctor:
        ANN ann = new ANN(2, 6, 3, 0.3, 42);

        // Training data: N samples, each with 2 inputs
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

        // One-hot labels: N samples, each with 3 outputs
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


        // Train (use a reasonable LR; 1.0 is usually too big for sigmoid+MSE)
        int epochs = 500000;
        ann.train(X, Y, epochs, 0.3);

        // Visualize
        JFrame f = new JFrame("3-Class Decision Boundary");
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.setContentPane(new BoundaryPanelMulti(ann, X, Y, 3));
        f.pack();
        f.setLocationRelativeTo(null);
        f.setVisible(true);

        // Test a few points
        test(ann, new double[]{0.08, 0.10}); // class 0-ish
        test(ann, new double[]{0.92, 0.08}); // class 1-ish
        test(ann, new double[]{0.05, 0.95}); // class 2-ish

        // In-between points
        test(ann, new double[]{0.60, 0.10});
        test(ann, new double[]{0.10, 0.60});
        test(ann, new double[]{0.35, 0.35});
    }

    private static void test(ANN ann, double[] x) {
        double[] out = ann.apply(x);
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
