package ch.innunvation.tools;

import ch.innunvation.GenderClassifier;
import ch.innunvation.ann.ANNCrossEntropy;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * Diagnostic tool to inspect your dataset
 */
public class DatasetDiagnostic {

    public static void main(String[] args) throws IOException {
        String dataDir = "src/main/resources/faces_dataset";

        System.out.println("=== Dataset Diagnostic ===\n");

        // Check male images
        System.out.println("--- MALE IMAGES ---");
        File maleDir = new File(dataDir, "male");
        analyzeDirectory(maleDir, "Male", 10);

        // Check female images
        System.out.println("\n--- FEMALE IMAGES ---");
        File femaleDir = new File(dataDir, "female");
        analyzeDirectory(femaleDir, "Female", 10);

        // Test preprocessing on sample images
        System.out.println("\n--- PREPROCESSING TEST ---");
        testPreprocessing(maleDir, femaleDir);

        // NEW: Test if network can learn at all with simple synthetic data
        System.out.println("\n--- NETWORK SANITY CHECK ---");
        testNetworkLearning();
    }

    /**
     * Test if the ANNCrossEntropy network can learn a simple pattern
     */
    private static void testNetworkLearning() {
        System.out.println("Testing if ANNCrossEntropy can learn XOR pattern...");

        // Simple XOR-like pattern
        double[][] X = {
                {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0},
                {0.1, 0.1}, {0.1, 0.9}, {0.9, 0.1}, {0.9, 0.9}
        };
        double[][] Y = {
                {0.0}, {1.0}, {1.0}, {0.0},
                {0.0}, {1.0}, {1.0}, {0.0}
        };

        ANNCrossEntropy network = new ANNCrossEntropy(2, 4, 1, 0.5, 42);

        System.out.println("Before training:");
        for (int i = 0; i < 4; i++) {
            double[] pred = network.apply(X[i]);
            System.out.printf("  Input [%.1f, %.1f] -> %.3f (expected: %.1f)%n",
                    X[i][0], X[i][1], pred[0], Y[i][0]);
        }

        // Train for 1000 epochs
        network.train(X, Y, 1000, 0.5);

        System.out.println("\nAfter 1000 epochs:");
        int correct = 0;
        for (int i = 0; i < 4; i++) {
            double[] pred = network.apply(X[i]);
            boolean isCorrect = Math.round(pred[0]) == Math.round(Y[i][0]);
            if (isCorrect) correct++;
            System.out.printf("  Input [%.1f, %.1f] -> %.3f (expected: %.1f) %s%n",
                    X[i][0], X[i][1], pred[0], Y[i][0], isCorrect ? "✓" : "✗");
        }

        if (correct >= 3) {
            System.out.println("✓ Network CAN learn! Problem is with your image data.");
        } else {
            System.out.println("✗ Network CANNOT learn! There's a bug in ANNCrossEntropy.");
        }
    }

    private static void analyzeDirectory(File dir, String label, int samplesToShow) throws IOException {
        if (!dir.exists()) {
            System.out.println("Directory not found: " + dir);
            return;
        }

        File[] files = dir.listFiles((d, name) -> {
            String n = name.toLowerCase();
            return n.endsWith(".jpg") || n.endsWith(".jpeg") || n.endsWith(".png");
        });

        if (files == null || files.length == 0) {
            System.out.println("No images found!");
            return;
        }

        System.out.println("Total images: " + files.length);
        System.out.println("Sample filenames:");

        for (int i = 0; i < Math.min(samplesToShow, files.length); i++) {
            File f = files[i];
            BufferedImage img = ImageIO.read(f);
            System.out.printf("  %s: %dx%d pixels%n",
                    f.getName(), img.getWidth(), img.getHeight());
        }
    }

    private static void testPreprocessing(File maleDir, File femaleDir) throws IOException {
        // Load one male and one female image
        File[] maleFiles = maleDir.listFiles((d, n) -> n.toLowerCase().endsWith(".jpg"));
        File[] femaleFiles = femaleDir.listFiles((d, n) -> n.toLowerCase().endsWith(".jpg"));

        if (maleFiles == null || maleFiles.length == 0 ||
                femaleFiles == null || femaleFiles.length == 0) {
            System.out.println("Could not find sample images");
            return;
        }

        BufferedImage maleImg = ImageIO.read(maleFiles[0]);
        BufferedImage femaleImg = ImageIO.read(femaleFiles[0]);

        // Preprocess
        double[] maleFeatures = GenderClassifier.preprocessImage(maleImg);
        double[] femaleFeatures = GenderClassifier.preprocessImage(femaleImg);

        // Check if features are all the same (indicates preprocessing problem)
        System.out.println("\nMale image: " + maleFiles[0].getName());
        System.out.println("  First 10 pixel values: " + formatArray(maleFeatures, 10));
        System.out.println("  Min value: " + min(maleFeatures));
        System.out.println("  Max value: " + max(maleFeatures));
        System.out.println("  Mean value: " + mean(maleFeatures));
        System.out.println("  Std dev: " + stdDev(maleFeatures));

        System.out.println("\nFemale image: " + femaleFiles[0].getName());
        System.out.println("  First 10 pixel values: " + formatArray(femaleFeatures, 10));
        System.out.println("  Min value: " + min(femaleFeatures));
        System.out.println("  Max value: " + max(femaleFeatures));
        System.out.println("  Mean value: " + mean(femaleFeatures));
        System.out.println("  Std dev: " + stdDev(femaleFeatures));

        // Warning if images are too similar
        double similarity = cosineSimilarity(maleFeatures, femaleFeatures);
        System.out.println("\nCosine similarity between male/female sample: " +
                String.format("%.4f", similarity));
        if (similarity > 0.95) {
            System.out.println("⚠️  WARNING: Images are very similar! This could indicate:");
            System.out.println("  - Images are not actually faces");
            System.out.println("  - Preprocessing is broken");
            System.out.println("  - Dataset has quality issues");
        }

        // Visual inspection
        showPreprocessedImages(maleImg, femaleImg, maleFeatures, femaleFeatures);
    }

    private static void showPreprocessedImages(BufferedImage origMale, BufferedImage origFemale,
                                               double[] maleFeat, double[] femaleFeat) {
        JFrame frame = new JFrame("Preprocessed Images - Check if faces are visible");
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        frame.setLayout(new GridLayout(2, 2));

        // Original images
        frame.add(new JLabel("Original Male",
                new ImageIcon(origMale.getScaledInstance(200, 200, Image.SCALE_SMOOTH)),
                SwingConstants.CENTER));
        frame.add(new JLabel("Original Female",
                new ImageIcon(origFemale.getScaledInstance(200, 200, Image.SCALE_SMOOTH)),
                SwingConstants.CENTER));

        // Preprocessed images
        BufferedImage maleProc = arrayToImage(maleFeat);
        BufferedImage femaleProc = arrayToImage(femaleFeat);
        frame.add(new JLabel("Preprocessed Male (48x48)",
                new ImageIcon(maleProc.getScaledInstance(200, 200, Image.SCALE_SMOOTH)),
                SwingConstants.CENTER));
        frame.add(new JLabel("Preprocessed Female (48x48)",
                new ImageIcon(femaleProc.getScaledInstance(200, 200, Image.SCALE_SMOOTH)),
                SwingConstants.CENTER));

        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);

        System.out.println("\n✓ Visual inspection window opened");
        System.out.println("  Check if preprocessed images show clear faces");
        System.out.println("  If images are blank/blurry/unrecognizable, preprocessing is broken");
    }

    private static BufferedImage arrayToImage(double[] pixels) {
        BufferedImage img = new BufferedImage(48, 48, BufferedImage.TYPE_BYTE_GRAY);
        for (int i = 0; i < pixels.length; i++) {
            int gray = (int) (pixels[i] * 255);
            int rgb = (gray << 16) | (gray << 8) | gray;
            img.setRGB(i % 48, i / 48, rgb);
        }
        return img;
    }

    private static String formatArray(double[] arr, int n) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < Math.min(n, arr.length); i++) {
            sb.append(String.format("%.3f", arr[i]));
            if (i < n - 1 && i < arr.length - 1) sb.append(", ");
        }
        sb.append("...]");
        return sb.toString();
    }

    private static double min(double[] arr) {
        double m = arr[0];
        for (double v : arr) if (v < m) m = v;
        return m;
    }

    private static double max(double[] arr) {
        double m = arr[0];
        for (double v : arr) if (v > m) m = v;
        return m;
    }

    private static double mean(double[] arr) {
        double sum = 0;
        for (double v : arr) sum += v;
        return sum / arr.length;
    }

    private static double stdDev(double[] arr) {
        double m = mean(arr);
        double sum = 0;
        for (double v : arr) sum += (v - m) * (v - m);
        return Math.sqrt(sum / arr.length);
    }

    private static double cosineSimilarity(double[] a, double[] b) {
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }
}