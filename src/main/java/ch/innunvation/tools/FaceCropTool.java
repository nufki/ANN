package ch.innunvation.tools;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * Simple center-crop tool for face images
 * Assumes face is roughly centered in the image
 */
public class FaceCropTool {

    /**
     * Crop to center square of image (removes background, keeps face)
     */
    public static BufferedImage centerCrop(BufferedImage img) {
        int width = img.getWidth();
        int height = img.getHeight();

        // Calculate square crop size (smaller dimension)
        int cropSize = Math.min(width, height);

        // Calculate crop coordinates (center)
        int x = (width - cropSize) / 2;
        int y = (height - cropSize) / 2;

        // Crop to square
        return img.getSubimage(x, y, cropSize, cropSize);
    }

    /**
     * Aggressive crop: takes center 60% of image
     * This removes more background and focuses on the face
     */
    public static BufferedImage aggressiveCrop(BufferedImage img) {
        int width = img.getWidth();
        int height = img.getHeight();

        // Crop to 60% of size
        int cropWidth = (int) (width * 0.6);
        int cropHeight = (int) (height * 0.6);
        int cropSize = Math.min(cropWidth, cropHeight);

        // Center the crop
        int x = (width - cropSize) / 2;
        int y = (height - cropSize) / 2;

        return img.getSubimage(x, y, cropSize, cropSize);
    }

    /**
     * Process all images in a directory
     */
    public static void processDirectory(File inputDir, File outputDir, boolean aggressive) throws IOException {
        if (!inputDir.exists() || !inputDir.isDirectory()) {
            System.err.println("Input directory doesn't exist: " + inputDir);
            return;
        }

        outputDir.mkdirs();

        File[] files = inputDir.listFiles((d, name) -> {
            String n = name.toLowerCase();
            return n.endsWith(".jpg") || n.endsWith(".jpeg") || n.endsWith(".png");
        });

        if (files == null) return;

        int processed = 0;
        for (File file : files) {
            try {
                BufferedImage img = ImageIO.read(file);
                BufferedImage cropped = aggressive ? aggressiveCrop(img) : centerCrop(img);

                // Resize to 48x48
                BufferedImage resized = new BufferedImage(48, 48, BufferedImage.TYPE_INT_RGB);
                Graphics2D g = resized.createGraphics();
                g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
                g.drawImage(cropped, 0, 0, 48, 48, null);
                g.dispose();

                // Save
                File outputFile = new File(outputDir, file.getName());
                ImageIO.write(resized, "jpg", outputFile);
                processed++;

                if (processed % 100 == 0) {
                    System.out.println("Processed " + processed + " images...");
                }
            } catch (Exception e) {
                System.err.println("Error processing " + file.getName() + ": " + e.getMessage());
            }
        }

        System.out.println("✓ Processed " + processed + " images");
    }

    public static void main(String[] args) throws IOException {
        String inputBase = "src/main/resources/faces_dataset";
        String outputBase = "src/main/resources/faces_dataset_cropped";

        System.out.println("=== Face Crop Tool ===");
        System.out.println("Input: " + inputBase);
        System.out.println("Output: " + outputBase);
        System.out.println();

        // Process male images
        System.out.println("Processing male images...");
        File maleInput = new File(inputBase, "male");
        File maleOutput = new File(outputBase, "male");
        processDirectory(maleInput, maleOutput, true); // aggressive crop

        // Process female images
        System.out.println("\nProcessing female images...");
        File femaleInput = new File(inputBase, "female");
        File femaleOutput = new File(outputBase, "female");
        processDirectory(femaleInput, femaleOutput, true); // aggressive crop

        System.out.println("\n✓ Done! New dataset created at: " + outputBase);
        System.out.println("Update GenderClassifier to use this new path:");
        System.out.println("  String dataDir = \"src/main/resources/faces_dataset_cropped\";");
    }
}