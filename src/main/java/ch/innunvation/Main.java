package ch.innunvation;

import java.util.Random;

public class Main {

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

        // 2 inputs, 3 hidden neurons, 1 output
        ANN ann = new ANN(2, 3, 1, new Random(42));

        // Train in one call
        int epochs = 50_000;
        double learningRate = 0.5;
        ann.train(X, Y, epochs, learningRate);

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
