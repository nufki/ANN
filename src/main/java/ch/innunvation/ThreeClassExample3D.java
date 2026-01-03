package ch.innunvation;

import ch.innunvation.ann.ANN;
import ch.innunvation.ui.BoundaryPanel3DTrue;

import javax.swing.*;
import java.util.Arrays;

public class ThreeClassExample3D {

    public static void main(String[] args) {
        // ANN with 3 inputs, hidden layer, and 3 outputs
        ANN ann = new ANN(3, 8, 3, 0.3, 42);

        // Training data: N samples, each with 3 inputs (x, y, z)
        double[][] X = {
                // Class 0 region (near origin: 0,0,0)
                {0.05, 0.05, 0.05},
                {0.10, 0.00, 0.08},
                {0.00, 0.15, 0.05},
                {0.12, 0.08, 0.10},
                {0.20, 0.10, 0.05},
                {0.08, 0.05, 0.15},
                {0.05, 0.12, 0.08},
                {0.15, 0.05, 0.05},

                // Class 1 region (near 1,0,0)
                {0.90, 0.05, 0.10},
                {1.00, 0.10, 0.05},
                {0.85, 0.00, 0.12},
                {0.95, 0.15, 0.08},
                {0.80, 0.10, 0.15},
                {0.92, 0.05, 0.05},
                {0.88, 0.12, 0.10},
                {0.85, 0.08, 0.08},

                // Class 2 region (near 0,1,1) - different corner
                {0.05, 0.90, 0.85},
                {0.10, 1.00, 0.90},
                {0.00, 0.85, 0.88},
                {0.15, 0.95, 0.92},
                {0.10, 0.80, 0.85},
                {0.08, 0.92, 0.90},
                {0.12, 0.88, 0.88},
                {0.05, 0.85, 0.95},
        };

        // One-hot labels: N samples, each with 3 outputs
        double[][] Y = {
                // Class 0 -> [1,0,0]
                {1, 0, 0},
                {1, 0, 0},
                {1, 0, 0},
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
                {0, 1, 0},
                {0, 1, 0},
                {0, 1, 0},

                // Class 2 -> [0,0,1]
                {0, 0, 1},
                {0, 0, 1},
                {0, 0, 1},
                {0, 0, 1},
                {0, 0, 1},
                {0, 0, 1},
                {0, 0, 1},
                {0, 0, 1},
        };

        // Train the network
        int epochs = 500000;
        System.out.println("Training 3-class 3D classifier...");
        System.out.println("Architecture: 3 inputs -> 8 hidden -> 3 outputs");
        System.out.println("Epochs: " + epochs);
        ann.train(X, Y, epochs, 0.3);
        System.out.println("Training completed!\n");

        // Visualize with true 3D panel (interactive rotation)
        JFrame f = new JFrame("3-Class Decision Boundary (3D - Interactive)");
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.setContentPane(new BoundaryPanel3DTrue(ann, X, Y, 3));
        f.pack();
        f.setLocationRelativeTo(null);
        f.setVisible(true);
        
        // Optional: Also show slice-based view in a second window
        // Uncomment to see both visualizations:
        /*
        JFrame f2 = new JFrame("3-Class Decision Boundary (3D - Slices)");
        f2.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        f2.setContentPane(new BoundaryPanel3D(ann, X, Y, 3));
        f2.pack();
        f2.setLocationRelativeTo(null);
        f2.setVisible(true);
        */

        // Test a few points
        System.out.println("=== Test Predictions ===");
        test(ann, new double[]{0.08, 0.10, 0.06}); // class 0-ish
        test(ann, new double[]{0.92, 0.08, 0.10}); // class 1-ish
        test(ann, new double[]{0.05, 0.95, 0.90}); // class 2-ish

        // In-between points
        test(ann, new double[]{0.50, 0.50, 0.50});
        test(ann, new double[]{0.30, 0.20, 0.40});
        test(ann, new double[]{0.70, 0.80, 0.60});
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

