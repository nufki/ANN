package ch.innunvation;

import javax.swing.*;
import java.util.Arrays;

public class ThreeClassExampleOverfitting {

    public static void main(String[] args) {
        // If your ANN has (inputs, hidden, outputs, learningRate, seed) ctor:
        ANN ann = new ANN(2, 6, 3, 0.3, 42);

        // --- OVERFITTING TRAIN SET (tiny + a few wrong labels) ---
        // Intuition:
        // - True class clusters are roughly near (0,0), (1,0), (0,1)
        // - But we give very few samples and include contradictions/outliers,
        //   so the network bends the boundary to "memorize" them.

        double[][] X = {
                // Class 0 (near 0,0) — only a few points
                {0.02, 0.01},
                {0.10, 0.05},
                {0.06, 0.12},

                // Class 1 (near 1,0) — only a few points
                {0.92, 0.03},
                {1.02, 0.06},
                {0.88, 0.10},

                // Class 2 (near 0,1) — only a few points
                {0.03, 0.92},
                {0.08, 1.02},
                {0.12, 0.86},

                // --- Contradictions / outliers (intentionally "wrong") ---
                // A point located in class-0-ish area but labeled as Class 1
                {0.08, 0.06},

                // A point located in class-1-ish area but labeled as Class 2
                {0.95, 0.08},

                // A point located in class-2-ish area but labeled as Class 0
                {0.05, 0.95},

                // A weird central point (forces boundary to do something odd)
                {0.45, 0.45},
        };

        double[][] Y = {
                // Class 0 -> [1,0,0]
                {1, 0, 0},
                {1, 0, 0},
                {1, 0, 0},

                // Class 1 -> [0,1,0]
                {0, 1, 0},
                {0, 1, 0},
                {0, 1, 0},

                // Class 2 -> [0,0,1]
                {0, 0, 1},
                {0, 0, 1},
                {0, 0, 1},

                // Contradictions (labels intentionally wrong for their location)
                {0, 1, 0}, // (0.08,0.06) forced to Class 1
                {0, 0, 1}, // (0.95,0.08) forced to Class 2
                {1, 0, 0}, // (0.05,0.95) forced to Class 0

                // Central point: pick a class (forces extra bending)
                {0, 1, 0}, // (0.45,0.45) forced to Class 1
        };

        // Train (use a reasonable LR; 1.0 is usually too big for sigmoid+MSE)
        int epochs = 50000;
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
