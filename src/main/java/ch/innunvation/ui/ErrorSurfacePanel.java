package ch.innunvation.ui;

import ch.innunvation.ann.ANN;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.util.ArrayList;
import java.util.List;

/**
 * 3D Error Surface Visualization Panel
 * Shows the error surface (MSE) as a function of two weights (w1 and w2).
 * The surface is visualized in 3D with interactive rotation.
 */
public class ErrorSurfacePanel extends JPanel {
    
    private final ANN originalANN;
    private final double[][] X;
    private final double[][] Y;
    
    // Which weights to visualize - configurable
    private final int hiddenIdx1, inputIdx1;  // First weight: w1[hiddenIdx1][inputIdx1]
    private final int hiddenIdx2, inputIdx2;  // Second weight: w1[hiddenIdx2][inputIdx2]
    private final String weightDescription;
    
    // 3D view configuration
    private final int plotSize = 400; // Larger for 3x2 grid (more space per plot)
    private final int padding = 0;   // More padding for better visibility
    
    // 3D rotation angles (in radians)
    private double rotationX = 0.6; // Slightly adjusted for better initial view
    private double rotationY = 0.4; // Slightly adjusted for better initial view
    private double zoom = 1.4; // Start with neutral zoom
    
    // Mouse interaction
    private int lastMouseX, lastMouseY;
    private boolean isDragging = false;
    
    // Error surface data
    private double[][] errorSurface;
    private double minW1, maxW1, minW2, maxW2;
    private double minError, maxError;
    private static final int SURFACE_RESOLUTION = 30; // Reduced resolution for smaller plots
    
    // Current (final/trained) weight values (for marking on surface)
    private double currentW1, currentW2;
    // Initial weight values (before training)
    private double initialW1, initialW2;
    // Training path (list of [w1, w2] pairs)
    private final List<double[]> trainingPath;
    
    /**
     * Constructor with default weights (w1[0][0] vs w1[0][1])
     */
    public ErrorSurfacePanel(ANN ann, double[][] X, double[][] Y) {
        this(ann, null, X, Y, 0, 0, 0, 1, "w1[0][0] vs w1[0][1]", null);
    }
    
    /**
     * Constructor with configurable weights (without initial ANN)
     */
    public ErrorSurfacePanel(ANN ann, double[][] X, double[][] Y,
                            int hiddenIdx1, int inputIdx1,
                            int hiddenIdx2, int inputIdx2,
                            String description) {
        this(ann, null, X, Y, hiddenIdx1, inputIdx1, hiddenIdx2, inputIdx2, description, null);
    }
    
    /**
     * Constructor with configurable weights and initial ANN for initial weight position
     * @param ann final/trained ANN
     * @param initialANN initial ANN (before training) - can be null
     * @param hiddenIdx1 hidden neuron index for first weight
     * @param inputIdx1 input index for first weight
     * @param hiddenIdx2 hidden neuron index for second weight
     * @param inputIdx2 input index for second weight
     * @param description description of the weight pair
     */
    public ErrorSurfacePanel(ANN ann, ANN initialANN, double[][] X, double[][] Y,
                            int hiddenIdx1, int inputIdx1,
                            int hiddenIdx2, int inputIdx2,
                            String description) {
        this(ann, initialANN, X, Y, hiddenIdx1, inputIdx1, hiddenIdx2, inputIdx2, description, null);
    }
    
    /**
     * Constructor with configurable weights, initial ANN, and training path
     * @param ann final/trained ANN
     * @param initialANN initial ANN (before training) - can be null
     * @param hiddenIdx1 hidden neuron index for first weight
     * @param inputIdx1 input index for first weight
     * @param hiddenIdx2 hidden neuron index for second weight
     * @param inputIdx2 input index for second weight
     * @param description description of the weight pair
     * @param trainingPath list of [w1, w2] pairs representing the training path
     */
    public ErrorSurfacePanel(ANN ann, ANN initialANN, double[][] X, double[][] Y,
                            int hiddenIdx1, int inputIdx1,
                            int hiddenIdx2, int inputIdx2,
                            String description, List<double[]> trainingPath) {
        this.originalANN = ann;
        this.X = X;
        this.Y = Y;
        this.hiddenIdx1 = hiddenIdx1;
        this.inputIdx1 = inputIdx1;
        this.hiddenIdx2 = hiddenIdx2;
        this.inputIdx2 = inputIdx2;
        this.weightDescription = description;
        this.trainingPath = trainingPath;
        
        // Get current (final/trained) weight values
        try {
            currentW1 = ann.getWeight(1, inputIdx1, hiddenIdx1);
            currentW2 = ann.getWeight(1, inputIdx2, hiddenIdx2);
        } catch (Exception e) {
            System.err.println("Error getting current weights: " + e.getMessage());
            currentW1 = 0.0;
            currentW2 = 0.0;
        }
        
        // Get initial weight values (before training)
        if (initialANN != null) {
            try {
                initialW1 = initialANN.getWeight(1, inputIdx1, hiddenIdx1);
                initialW2 = initialANN.getWeight(1, inputIdx2, hiddenIdx2);
            } catch (Exception e) {
                System.err.println("Error getting initial weights: " + e.getMessage());
                initialW1 = currentW1; // Fallback to current if can't get initial
                initialW2 = currentW2;
            }
        } else {
            // No initial ANN provided, use current weights as initial
            initialW1 = currentW1;
            initialW2 = currentW2;
        }
        
        // Set preferred size so layout manager knows how big this component should be
        // Account for full plot area including padding
        int totalWidth = padding * 2 + plotSize;
        int totalHeight = padding * 2 + plotSize; // No extra space needed - title is in border
        setPreferredSize(new Dimension(totalWidth, totalHeight));
        setMinimumSize(new Dimension(totalWidth, totalHeight));
        setOpaque(true); // Ensure background is painted
        setBackground(Color.WHITE);
        
        // Initialize error surface as empty initially
        errorSurface = new double[SURFACE_RESOLUTION][SURFACE_RESOLUTION];
        minError = 0;
        maxError = 1;
        
        // Compute error surface - do it synchronously first time to ensure it works
        // (can be slow but ensures we see something)
        try {
            computeErrorSurface();
            System.out.println("Error surface computed. MinErr: " + minError + " MaxErr: " + maxError);
        } catch (Exception e) {
            System.err.println("Error computing surface: " + e.getMessage());
            e.printStackTrace();
            // Fill with dummy data so something shows
            for (int i = 0; i < SURFACE_RESOLUTION; i++) {
                for (int j = 0; j < SURFACE_RESOLUTION; j++) {
                    errorSurface[i][j] = 0.1 + 0.1 * Math.sin(i * 0.2) * Math.cos(j * 0.2);
                }
            }
            minError = 0.0;
            maxError = 0.3;
        }
        
        setupMouseControls();
        
        // Force immediate repaint
        repaint();
    }
    
    private void computeErrorSurface() {
        // Determine weight ranges based on both initial and current weights
        // to ensure both positions are visible
        double range = 2.0; // Range around weights
        double minW1Initial = initialW1 - range;
        double maxW1Initial = initialW1 + range;
        double minW2Initial = initialW2 - range;
        double maxW2Initial = initialW2 + range;
        
        double minW1Current = currentW1 - range;
        double maxW1Current = currentW1 + range;
        double minW2Current = currentW2 - range;
        double maxW2Current = currentW2 + range;
        
        // Use the union of ranges to include both initial and final positions
        minW1 = Math.min(minW1Initial, minW1Current);
        maxW1 = Math.max(maxW1Initial, maxW1Current);
        minW2 = Math.min(minW2Initial, minW2Current);
        maxW2 = Math.max(maxW2Initial, maxW2Current);
        
        // Ensure valid ranges
        if (maxW1 <= minW1) {
            minW1 = -2.0;
            maxW1 = 2.0;
        }
        if (maxW2 <= minW2) {
            minW2 = -2.0;
            maxW2 = 2.0;
        }
        
        errorSurface = new double[SURFACE_RESOLUTION][SURFACE_RESOLUTION];
        minError = Double.POSITIVE_INFINITY;
        maxError = Double.NEGATIVE_INFINITY;
        
        System.out.println("Computing error surface for " + weightDescription);
        System.out.println("Weight ranges: W1=[" + minW1 + ", " + maxW1 + "], W2=[" + minW2 + ", " + maxW2 + "]");
        System.out.println("Training data size: " + X.length + " samples");
        
        // Compute error for each weight combination
        int computed = 0;
        for (int i = 0; i < SURFACE_RESOLUTION; i++) {
            double w1 = minW1 + (maxW1 - minW1) * i / (SURFACE_RESOLUTION - 1.0);
            for (int j = 0; j < SURFACE_RESOLUTION; j++) {
                double w2 = minW2 + (maxW2 - minW2) * j / (SURFACE_RESOLUTION - 1.0);
                
                try {
                    double error = computeErrorForWeights(w1, w2);
                    errorSurface[i][j] = error;
                    
                    if (Double.isFinite(error)) {
                        minError = Math.min(minError, error);
                        maxError = Math.max(maxError, error);
                        computed++;
                    }
                } catch (Exception e) {
                    System.err.println("Error computing error for w1=" + w1 + ", w2=" + w2 + ": " + e.getMessage());
                    errorSurface[i][j] = Double.MAX_VALUE;
                }
            }
        }
        
        System.out.println("Computed " + computed + " valid error values");
        System.out.println("Error range: [" + minError + ", " + maxError + "]");
        
        // Ensure we have a valid error range
        if (!Double.isFinite(minError) || !Double.isFinite(maxError) || maxError <= minError) {
            System.err.println("Invalid error range, using defaults");
            minError = 0.0;
            maxError = 1.0;
        }
    }
    
    private double computeErrorForWeights(double w1, double w2) {
        // Use the new method to compute error with modified weights
        // This is a 2D slice of the full error surface (all other weights are held constant)
        return originalANN.mseWithTwoModifiedWeights(X, Y, 
                hiddenIdx1, inputIdx1, w1,
                hiddenIdx2, inputIdx2, w2);
    }
    
    /**
     * Gets information about which weights are being visualized
     */
    public String getVisualizedWeightsInfo() {
        return String.format("Visualizing: %s\n" +
                "This is a 2D slice of the full %d-dimensional weight space.\n" +
                "All other weights are held constant at their trained values.",
                weightDescription, getTotalParameters());
    }
    
    /**
     * Gets the weight description
     */
    public String getWeightDescription() {
        return weightDescription;
    }
    
    private int getTotalParameters() {
        // Calculate total parameters: (inputs * hidden) + hidden + (hidden * outputs) + outputs
        // For 2 inputs, 6 hidden, 3 outputs: (2*6) + 6 + (6*3) + 3 = 12 + 6 + 18 + 3 = 39
        // But we need to get this from the ANN
        try {
            java.lang.reflect.Field nInField = ANN.class.getDeclaredField("nIn");
            java.lang.reflect.Field nHiddenField = ANN.class.getDeclaredField("nHidden");
            java.lang.reflect.Field nOutField = ANN.class.getDeclaredField("nOut");
            nInField.setAccessible(true);
            nHiddenField.setAccessible(true);
            nOutField.setAccessible(true);
            
            int nIn = nInField.getInt(originalANN);
            int nHidden = nHiddenField.getInt(originalANN);
            int nOut = nOutField.getInt(originalANN);
            
            return (nIn * nHidden) + nHidden + (nHidden * nOut) + nOut;
        } catch (Exception e) {
            return 25; // Default estimate
        }
    }
    
    private void setupMouseControls() {
        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                if (e.getButton() == MouseEvent.BUTTON1) {
                    isDragging = true;
                    lastMouseX = e.getX();
                    lastMouseY = e.getY();
                }
            }
            
            @Override
            public void mouseReleased(MouseEvent e) {
                isDragging = false;
            }
        });
        
        addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                if (isDragging) {
                    int dx = e.getX() - lastMouseX;
                    int dy = e.getY() - lastMouseY;
                    
                    rotationY += dx * 0.01;
                    rotationX += dy * 0.01;
                    
                    // Clamp rotationX
                    rotationX = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, rotationX));
                    
                    lastMouseX = e.getX();
                    lastMouseY = e.getY();
                    
                    repaint();
                }
            }
        });
        
        addMouseWheelListener(e -> {
            double zoomFactor = 1.0 + e.getWheelRotation() * 0.1;
            zoom *= zoomFactor;
            zoom = Math.max(0.5, Math.min(3.0, zoom));
            repaint();
        });
    }
    
    @Override
    protected void paintComponent(Graphics g0) {
        super.paintComponent(g0);
        
        Graphics2D g = (Graphics2D) g0.create();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        
        // Set clip to ensure nothing is drawn outside the component bounds
        g.setClip(0, 0, getWidth(), getHeight());
        
        // Draw plot area background (white, no border)
        g.setColor(Color.WHITE);
        g.fillRect(padding, padding, plotSize, plotSize);
        
        // Title is drawn by the TitledBorder in the container, so we don't draw it here
        
        // Check if error surface is computed
        if (errorSurface == null || errorSurface.length == 0) {
            g.setColor(Color.BLUE);
            g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 12));
            g.drawString("Error: Surface is null", padding + 10, padding + 20);
            g.dispose();
            return;
        }
        
        // Check if surface is still being computed (all zeros or invalid)
        if (minError == 0 && maxError == 1 && errorSurface[0][0] == 0) {
            g.setColor(Color.BLUE);
            g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 12));
            g.drawString("Computing error surface...", padding + 10, padding + 20);
            g.dispose();
            return;
        }
        
        // Draw 3D error surface
        drawErrorSurface(g);
        
        // Draw 2D contour as very subtle overlay (optional, can be removed)
        // draw2DContourFallback(g);
        
        // Draw axes and labels
        drawAxes(g);
        
        // Draw instructions
        //drawInstructions(g);
        
        g.dispose();
    }
    
    private void drawErrorSurface(Graphics2D g) {
        // Project 3D points to 2D screen coordinates
        Point3D[][] surfacePoints = new Point3D[SURFACE_RESOLUTION][SURFACE_RESOLUTION];
        
        for (int i = 0; i < SURFACE_RESOLUTION; i++) {
            for (int j = 0; j < SURFACE_RESOLUTION; j++) {
                double w1 = minW1 + (maxW1 - minW1) * i / (SURFACE_RESOLUTION - 1.0);
                double w2 = minW2 + (maxW2 - minW2) * j / (SURFACE_RESOLUTION - 1.0);
                double error = errorSurface[i][j];
                
                // Normalize error for visualization
                double normalizedError = (error - minError) / (maxError - minError + 1e-9);
                
                Point3D p = project3D(w1, w2, normalizedError);
                surfacePoints[i][j] = p;
            }
        }
        
        // Draw surface as wireframe
        g.setStroke(new BasicStroke(1.0f));
        
        // Set clip to plot area to prevent drawing outside bounds
        g.setClip(padding, padding, plotSize, plotSize);
        
        for (int i = 0; i < SURFACE_RESOLUTION - 1; i++) {
            for (int j = 0; j < SURFACE_RESOLUTION - 1; j++) {
                Point3D p1 = surfacePoints[i][j];
                Point3D p2 = surfacePoints[i + 1][j];
                Point3D p3 = surfacePoints[i][j + 1];
                
                // Color based on error height
                double error = errorSurface[i][j];
                Color c = getErrorColor(error);
                g.setColor(c);
                
                // Draw lines - check for valid numbers and bounds
                if (Double.isFinite(p1.x) && Double.isFinite(p1.y) && 
                    Double.isFinite(p2.x) && Double.isFinite(p2.y)) {
                    // Only draw if both points are within reasonable bounds
                    if (p1.x >= padding - 100 && p1.x <= padding + plotSize + 100 &&
                        p1.y >= padding - 100 && p1.y <= padding + plotSize + 100 &&
                        p2.x >= padding - 100 && p2.x <= padding + plotSize + 100 &&
                        p2.y >= padding - 100 && p2.y <= padding + plotSize + 100) {
                        g.drawLine((int) p1.x, (int) p1.y, (int) p2.x, (int) p2.y);
                    }
                }
                if (Double.isFinite(p1.x) && Double.isFinite(p1.y) && 
                    Double.isFinite(p3.x) && Double.isFinite(p3.y)) {
                    // Only draw if both points are within reasonable bounds
                    if (p1.x >= padding - 100 && p1.x <= padding + plotSize + 100 &&
                        p1.y >= padding - 100 && p1.y <= padding + plotSize + 100 &&
                        p3.x >= padding - 100 && p3.x <= padding + plotSize + 100 &&
                        p3.y >= padding - 100 && p3.y <= padding + plotSize + 100) {
                        g.drawLine((int) p1.x, (int) p1.y, (int) p3.x, (int) p3.y);
                    }
                }
            }
        }
        
        // Reset clip for axes and other elements
        g.setClip(null);
        
        // Draw training path in black (if available)
        if (trainingPath != null && trainingPath.size() > 1) {
            drawTrainingPath(g);
        }
        
        // Mark initial weight position (before training) with black dot
        double initialError = computeErrorForWeights(initialW1, initialW2);
        double normalizedInitialError = (initialError - minError) / (maxError - minError + 1e-9);
        Point3D initialPoint = project3D(initialW1, initialW2, normalizedInitialError);
        if (initialPoint.x >= -50 && initialPoint.x < plotSize + 2 * padding + 50 &&
            initialPoint.y >= -50 && initialPoint.y < plotSize + 2 * padding + 50) {
            g.setColor(Color.BLACK);
            g.fillOval((int) initialPoint.x - 5, (int) initialPoint.y - 5, 10, 10);
            g.setColor(Color.WHITE);
            g.drawOval((int) initialPoint.x - 5, (int) initialPoint.y - 5, 10, 10);
        }
        
        // Mark final weight position (after training) with red dot
        double currentError = computeErrorForWeights(currentW1, currentW2);
        double normalizedError = (currentError - minError) / (maxError - minError + 1e-9);
        Point3D currentPoint = project3D(currentW1, currentW2, normalizedError);
        // Always try to draw the current point, even if marked as not visible
        if (currentPoint.x >= -50 && currentPoint.x < plotSize + 2 * padding + 50 &&
            currentPoint.y >= -50 && currentPoint.y < plotSize + 2 * padding + 50) {
            g.setColor(Color.RED);
            g.fillOval((int) currentPoint.x - 5, (int) currentPoint.y - 5, 10, 10);
            g.setColor(Color.WHITE);
            g.drawOval((int) currentPoint.x - 5, (int) currentPoint.y - 5, 10, 10);
        }
    }
    
    private void drawTrainingPath(Graphics2D g) {
        if (trainingPath == null || trainingPath.size() < 2) {
            return;
        }
        
        // Set clip to plot area
        g.setClip(padding, padding, plotSize, plotSize);
        
        // Draw path in black
        g.setColor(Color.BLACK);
        g.setStroke(new BasicStroke(2.0f));
        
        // Project all path points to 3D space
        List<Point3D> projectedPoints = new ArrayList<>();
        for (double[] point : trainingPath) {
            double w1 = point[0];
            double w2 = point[1];
            double error = computeErrorForWeights(w1, w2);
            double normalizedError = (error - minError) / (maxError - minError + 1e-9);
            Point3D p = project3D(w1, w2, normalizedError);
            projectedPoints.add(p);
        }
        
        // Draw lines connecting consecutive points
        for (int i = 0; i < projectedPoints.size() - 1; i++) {
            Point3D p1 = projectedPoints.get(i);
            Point3D p2 = projectedPoints.get(i + 1);
            
            // Only draw if both points are valid and visible
            if (Double.isFinite(p1.x) && Double.isFinite(p1.y) &&
                Double.isFinite(p2.x) && Double.isFinite(p2.y)) {
                // Draw line if both points are within reasonable bounds
                if (p1.x >= padding - 100 && p1.x <= padding + plotSize + 100 &&
                    p1.y >= padding - 100 && p1.y <= padding + plotSize + 100 &&
                    p2.x >= padding - 100 && p2.x <= padding + plotSize + 100 &&
                    p2.y >= padding - 100 && p2.y <= padding + plotSize + 100) {
                    g.drawLine((int) p1.x, (int) p1.y, (int) p2.x, (int) p2.y);
                }
            }
        }
        
        // Reset clip
        g.setClip(null);
    }
    
    private Color getErrorColor(double error) {
        // Map error to color: low error = green, high error = red
        double normalized = (error - minError) / (maxError - minError + 1e-9);
        normalized = Math.max(0, Math.min(1, normalized));
        
        int r = (int) (normalized * 255);
        int g = (int) ((1 - normalized) * 255);
        int b = 0;
        
        return new Color(r, g, b);
    }
    
    private Point3D project3D(double x, double y, double z) {
        // Center and scale weights to [-1, 1] range
        double centerX = (minW1 + maxW1) / 2.0;
        double centerY = (minW2 + maxW2) / 2.0;
        double centerZ = 0.5;
        
        double scaleW = Math.max(maxW1 - minW1, maxW2 - minW2);
        if (scaleW < 1e-9) scaleW = 1.0; // Avoid division by zero
        
        // Normalize weights to [-1, 1]
        double nx = (x - centerX) / scaleW;
        double ny = (y - centerY) / scaleW;
        
        // z is expected to be normalized [0, 1] for error
        // If z > 1, it's a raw error value, normalize it
        if (z > 1.0) {
            z = (z - minError) / (maxError - minError + 1e-9);
        }
        double nz = z - centerZ; // Now in [-0.5, 0.5] range
        
        // Apply rotation
        double cosX = Math.cos(rotationX);
        double sinX = Math.sin(rotationX);
        double cosY = Math.cos(rotationY);
        double sinY = Math.sin(rotationY);
        
        // Rotate around X axis
        double z1 = ny * sinX + nz * cosX;
        
        // Rotate around Y axis  
        double x2 = nx * cosY + z1 * sinY;
        double z2 = -nx * sinY + z1 * cosY;
        
        // Project to 2D (orthographic) - center in plot area with better scaling
        // Use smaller scale factor to ensure entire surface fits within plot bounds
        double scaleFactor = 0.3; // Scale to fit and center in plot, ensuring nothing is clipped
        double screenX = padding + plotSize / 2.0 + x2 * plotSize * zoom * scaleFactor;
        double screenY = padding + plotSize / 2.0 - z2 * plotSize * zoom * scaleFactor;
        
        // Visibility check - very lenient
        boolean visible = z2 > -10.0; // Show almost everything
        
        return new Point3D(screenX, screenY, z2, visible);
    }
    
    private void drawAxes(Graphics2D g) {
        // Only draw axes if surface is computed
        if (errorSurface == null || minError == 0 && maxError == 1 && errorSurface[0][0] == 0) {
            return;
        }
        
        g.setColor(Color.DARK_GRAY);
        g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 9)); // Even smaller font
        g.setStroke(new BasicStroke(1.5f));
        
        // Project origin (0, 0, 0) to screen coordinates
        double centerW1 = (minW1 + maxW1) / 2.0;
        double centerW2 = (minW2 + maxW2) / 2.0;
        Point3D origin = project3D(centerW1, centerW2, 0.0);
        
        // Check if origin is valid (finite numbers)
        if (!Double.isFinite(origin.x) || !Double.isFinite(origin.y)) {
            return;
        }
        
        // Axis length in 3D space - smaller for compact plots
        double axisLength = Math.min(maxW1 - minW1, maxW2 - minW2) * 0.25;
        
        // X-axis (Weight 1) - blue
        Point3D xEnd = project3D(centerW1 + axisLength, centerW2, 0.0);
        // Always try to draw X-axis, even if slightly outside bounds
        if (Double.isFinite(xEnd.x) && Double.isFinite(xEnd.y) && 
            Double.isFinite(origin.x) && Double.isFinite(origin.y)) {
            g.setColor(Color.BLUE);
            g.drawLine((int) origin.x, (int) origin.y, (int) xEnd.x, (int) xEnd.y);
            drawArrow(g, (int) origin.x, (int) origin.y, (int) xEnd.x, (int) xEnd.y);
            // Only draw label if it's reasonably within bounds
            if (xEnd.x >= -100 && xEnd.x < plotSize + 2 * padding + 100 &&
                xEnd.y >= -100 && xEnd.y < plotSize + 2 * padding + 100) {
                g.drawString("W1", (int) xEnd.x + 3, (int) xEnd.y);
            }
        }
        
        // Y-axis (Weight 2) - blue
        Point3D yEnd = project3D(centerW1, centerW2 + axisLength, 0.0);
        // Always try to draw Y-axis, even if slightly outside bounds
        if (Double.isFinite(yEnd.x) && Double.isFinite(yEnd.y) && 
            Double.isFinite(origin.x) && Double.isFinite(origin.y)) {
            g.setColor(Color.BLUE);
            g.drawLine((int) origin.x, (int) origin.y, (int) yEnd.x, (int) yEnd.y);
            drawArrow(g, (int) origin.x, (int) origin.y, (int) yEnd.x, (int) yEnd.y);
            // Only draw label if it's reasonably within bounds
            if (yEnd.x >= -100 && yEnd.x < plotSize + 2 * padding + 100 &&
                yEnd.y >= -100 && yEnd.y < plotSize + 2 * padding + 100) {
                g.drawString("W2", (int) yEnd.x + 3, (int) yEnd.y);
            }
        }
        
        // Z-axis (Error) - red
        double normalizedMaxError = 1.0;
        Point3D zEnd = project3D(centerW1, centerW2, normalizedMaxError);
        // Always try to draw Z-axis, even if slightly outside bounds
        if (Double.isFinite(zEnd.x) && Double.isFinite(zEnd.y) && 
            Double.isFinite(origin.x) && Double.isFinite(origin.y)) {
            g.setColor(Color.RED);
            g.drawLine((int) origin.x, (int) origin.y, (int) zEnd.x, (int) zEnd.y);
            drawArrow(g, (int) origin.x, (int) origin.y, (int) zEnd.x, (int) zEnd.y);
            // Only draw label if it's reasonably within bounds
            if (zEnd.x >= -100 && zEnd.x < plotSize + 2 * padding + 100 &&
                zEnd.y >= -100 && zEnd.y < plotSize + 2 * padding + 100) {
                g.drawString("Err", (int) zEnd.x + 3, (int) zEnd.y - 8);
            }
        }
    }
    
    private void drawArrow(Graphics2D g, int x1, int y1, int x2, int y2) {
        double angle = Math.atan2(y2 - y1, x2 - x1);
        int arrowLength = 10;
        
        // Draw arrow head
        double x3 = x2 - arrowLength * Math.cos(angle - Math.PI / 6);
        double y3 = y2 - arrowLength * Math.sin(angle - Math.PI / 6);
        double x4 = x2 - arrowLength * Math.cos(angle + Math.PI / 6);
        double y4 = y2 - arrowLength * Math.sin(angle + Math.PI / 6);
        
        int[] xPoints = {x2, (int) x3, (int) x4};
        int[] yPoints = {y2, (int) y3, (int) y4};
        g.fillPolygon(xPoints, yPoints, 3);
    }
    
    private void drawInstructions(Graphics2D g) {
        g.setColor(Color.BLACK);
        g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 9)); // Smaller font
        
        String[] instructions = {
            "Drag: rotate",
            "Scroll: zoom",
            "Red: current"
        };
        
        int y = padding + 12;
        for (String inst : instructions) {
            g.drawString(inst, padding + 3, y);
            y += 10;
        }
    }
    
    /**
     * Draw a simple 2D contour plot as fallback when 3D projection fails
     */
    private void draw2DContourFallback(Graphics2D g) {
        if (errorSurface == null || errorSurface.length == 0) {
            g.setColor(Color.RED);
            g.drawString("No surface data", padding + 10, padding + 20);
            return;
        }
        
        // Draw error as a semi-transparent 2D heatmap/contour overlay
        int cellW = Math.max(1, plotSize / SURFACE_RESOLUTION);
        int cellH = Math.max(1, plotSize / SURFACE_RESOLUTION);
        
        // Use semi-transparent colors for overlay effect
        for (int i = 0; i < SURFACE_RESOLUTION; i++) {
            for (int j = 0; j < SURFACE_RESOLUTION; j++) {
                double error = errorSurface[i][j];
                if (!Double.isFinite(error)) continue;
                
                Color c = getErrorColor(error);
                // Make semi-transparent for overlay (allows 3D wireframe to show through)
                Color transparent = new Color(c.getRed(), c.getGreen(), c.getBlue(), 120);
                g.setColor(transparent);
                
                int x = padding + i * cellW;
                int y = padding + j * cellH;
                g.fillRect(x, y, cellW, cellH);
            }
        }
        
        // Mark current position
        if (maxW1 > minW1 && maxW2 > minW2) {
            double w1Norm = (currentW1 - minW1) / (maxW1 - minW1);
            double w2Norm = (currentW2 - minW2) / (maxW2 - minW2);
            w1Norm = Math.max(0, Math.min(1, w1Norm)); // Clamp
            w2Norm = Math.max(0, Math.min(1, w2Norm)); // Clamp
            int px = padding + (int)(w1Norm * plotSize);
            int py = padding + (int)(w2Norm * plotSize);
            g.setColor(Color.RED);
            g.fillOval(px - 5, py - 5, 10, 10);
            g.setColor(Color.WHITE);
            g.setStroke(new BasicStroke(2.0f));
            g.drawOval(px - 5, py - 5, 10, 10);
        }
        
        // Draw axis labels
        g.setColor(Color.BLACK);
        g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 9));
        g.drawString("W1", padding + plotSize / 2 - 10, padding + plotSize + 15);
        g.rotate(-Math.PI / 2);
        g.drawString("W2", -(padding + plotSize / 2), padding - 5);
        g.rotate(Math.PI / 2);
    }
    
    private static class Point3D {
        double x, y, z;
        boolean visible;
        
        Point3D(double x, double y, double z, boolean visible) {
            this.x = x;
            this.y = y;
            this.z = z;
            this.visible = visible;
        }
    }
}   