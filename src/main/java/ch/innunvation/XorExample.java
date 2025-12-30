package ch.innunvation;

import javax.swing.*;
import java.util.Random;

public class XorExample {

    public static void main(String[] args) {

        // XOR dataset
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

        int[] y = {0, 1, 1, 0};

        // 2 inputs, 3 hidden neurons, 1 output
        ANN ann = new ANN(2, 3, 1, new Random(42));

        // Train in one call
        int epochs = 50_000;
        double learningRate = 0.5;
        ann.train(X, Y, epochs, learningRate); // add custom learning rate

        // Visualize
        BoundaryPanelBinary panel = new BoundaryPanelBinary(
                ann, X, y,
                -0.25, 1.25,   // x-range
                -0.25, 1.25    // y-range
        );
        panel.rebuildBackground();

        SwingUtilities.invokeLater(() -> {
            JFrame f = new JFrame("Decision Boundary (XOR)");
            f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
            f.setContentPane(panel);
            f.pack();
            f.setLocationRelativeTo(null);
            f.setVisible(true);
        });

        // Evaluate
        for (double[] x : X) {
            double[] out = ann.apply(x);
            System.out.printf(
                    "Input (%.0f, %.0f) -> %.4f%n",
                    x[0], x[1], out[0]
            );
        }

        // Optional final error
        System.out.println("Final MSE = " + ann.mse(X, Y));
    }
}
