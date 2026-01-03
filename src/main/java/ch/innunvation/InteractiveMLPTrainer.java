package ch.innunvation;

import ch.innunvation.ann.ANN;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/**
 * Interactive MLP Trainer GUI
 * Allows users to:
 * - Add training data points by clicking on the scene
 * - Select which class (0, 1, or 2) to add points for
 * - Configure MLP parameters (hidden neurons, learning rate)
 * - Train the network and visualize decision boundaries
 */
public class InteractiveMLPTrainer extends JFrame {
    
    private static final int NUM_CLASSES = 3;
    private static final int DEFAULT_HIDDEN_NEURONS = 6;
    private static final double DEFAULT_LEARNING_RATE = 0.3;
    private static final int DEFAULT_EPOCHS = 50000;
    
    // Data storage - keep as arrays for better performance
    private final List<double[]> trainingData = new ArrayList<>();
    private final List<double[]> trainingLabels = new ArrayList<>();
    private double[][] XArray = new double[0][];
    private double[][] YArray = new double[0][];
    
    // UI Components
    private final InteractiveBoundaryPanel plotPanel;
    private final JButton[] classButtons = new JButton[NUM_CLASSES];
    private JTextField hiddenNeuronsField;
    private JTextField learningRateField;
    private JTextField epochsField;
    private JButton trainButton;
    private JLabel statusLabel;
    
    // Current selected class
    private int selectedClass = 0;
    
    // MLP instance
    private ANN ann = null;
    
    public InteractiveMLPTrainer() {
        setTitle("Interactive MLP Trainer - 3 Classes");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout());
        
        // Create main plot panel
        plotPanel = new InteractiveBoundaryPanel();
        add(plotPanel, BorderLayout.CENTER);
        
        // Create control panel
        JPanel controlPanel = createControlPanel();
        add(controlPanel, BorderLayout.SOUTH);
        
        pack();
        setLocationRelativeTo(null);
        setVisible(true);
    }
    
    private JPanel createControlPanel() {
        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
        panel.setBorder(BorderFactory.createTitledBorder("Configuration"));
        
        // Class selection buttons
        JPanel classPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        classPanel.setBorder(BorderFactory.createTitledBorder("Select Class"));
        for (int i = 0; i < NUM_CLASSES; i++) {
            final int classIdx = i;
            classButtons[i] = new JButton("Class " + i);
            classButtons[i].setPreferredSize(new Dimension(100, 30));
            classButtons[i].setOpaque(true);
            classButtons[i].setContentAreaFilled(true);
            classButtons[i].addActionListener(e -> {
                selectedClass = classIdx;
                updateClassButtonSelection();
            });
            classPanel.add(classButtons[i]);
        }
        updateClassButtonSelection();
        panel.add(classPanel);
        
        // MLP Configuration
        JPanel configPanel = new JPanel(new GridBagLayout());
        configPanel.setBorder(BorderFactory.createTitledBorder("MLP Configuration"));
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);
        gbc.anchor = GridBagConstraints.WEST;
        
        // Hidden neurons
        gbc.gridx = 0; gbc.gridy = 0;
        configPanel.add(new JLabel("Hidden Neurons:"), gbc);
        gbc.gridx = 1;
        hiddenNeuronsField = new JTextField(String.valueOf(DEFAULT_HIDDEN_NEURONS), 10);
        configPanel.add(hiddenNeuronsField, gbc);
        
        // Learning rate
        gbc.gridx = 0; gbc.gridy = 1;
        configPanel.add(new JLabel("Learning Rate:"), gbc);
        gbc.gridx = 1;
        learningRateField = new JTextField(String.valueOf(DEFAULT_LEARNING_RATE), 10);
        configPanel.add(learningRateField, gbc);
        
        // Epochs
        gbc.gridx = 0; gbc.gridy = 2;
        configPanel.add(new JLabel("Epochs:"), gbc);
        gbc.gridx = 1;
        epochsField = new JTextField(String.valueOf(DEFAULT_EPOCHS), 10);
        configPanel.add(epochsField, gbc);
        
        panel.add(configPanel);
        
        // Action buttons
        JPanel actionPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));
        trainButton = new JButton("Train");
        trainButton.setPreferredSize(new Dimension(120, 35));
        trainButton.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 14));
        trainButton.addActionListener(e -> trainNetwork());
        actionPanel.add(trainButton);

        JButton clearButton = new JButton("Clear Data");
        clearButton.setPreferredSize(new Dimension(120, 35));
        clearButton.addActionListener(e -> clearData());
        actionPanel.add(clearButton);
        
        panel.add(actionPanel);
        
        // Status label
        statusLabel = new JLabel("Click on the plot to add training data points");
        statusLabel.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
        panel.add(statusLabel);
        
        return panel;
    }
    
    private void updateClassButtonSelection() {
        // Color palette matching the point colors
        Color[] classColors = {
            new Color(200, 0, 0),      // Class 0 - Red
            new Color(0, 70, 200),      // Class 1 - Blue
            new Color(0, 140, 0)        // Class 2 - Green
        };
        
        for (int i = 0; i < NUM_CLASSES; i++) {
            if (i == selectedClass) {
                // Selected: thick colored border
                classButtons[i].setBorder(BorderFactory.createCompoundBorder(
                    BorderFactory.createLineBorder(classColors[i], 3),
                    BorderFactory.createEmptyBorder(2, 8, 2, 8)
                ));
            } else {
                // Not selected: default border
                classButtons[i].setBorder(BorderFactory.createCompoundBorder(
                    BorderFactory.createLineBorder(Color.GRAY, 1),
                    BorderFactory.createEmptyBorder(2, 8, 2, 8)
                ));
            }
        }
    }
    
    private void addDataPoint(double x, double y) {
        trainingData.add(new double[]{x, y});
        
        // Create one-hot encoding for the selected class
        double[] label = new double[NUM_CLASSES];
        label[selectedClass] = 1.0;
        trainingLabels.add(label);
        
        // Update arrays efficiently - only convert when needed for display
        XArray = trainingData.toArray(new double[trainingData.size()][]);
        YArray = trainingLabels.toArray(new double[trainingLabels.size()][]);
        
        plotPanel.setTrainingData(XArray, YArray);
        
        statusLabel.setText(String.format("Added Class %d point at (%.2f, %.2f). Total points: %d",
            selectedClass, x, y, trainingData.size()));
    }
    
    private void trainNetwork() {
        if (trainingData.isEmpty()) {
            JOptionPane.showMessageDialog(this,
                "Please add at least one training data point before training.",
                "No Data", JOptionPane.WARNING_MESSAGE);
            return;
        }
        
        // Parse configuration
        int hiddenNeurons;
        double learningRate;
        int epochs;
        
        try {
            hiddenNeurons = Integer.parseInt(hiddenNeuronsField.getText().trim());
            if (hiddenNeurons <= 0) {
                throw new NumberFormatException("Hidden neurons must be positive");
            }
        } catch (NumberFormatException e) {
            JOptionPane.showMessageDialog(this,
                "Invalid hidden neurons value. Please enter a positive integer.",
                "Invalid Input", JOptionPane.ERROR_MESSAGE);
            return;
        }
        
        try {
            learningRate = Double.parseDouble(learningRateField.getText().trim());
            if (learningRate <= 0) {
                throw new NumberFormatException("Learning rate must be positive");
            }
        } catch (NumberFormatException e) {
            JOptionPane.showMessageDialog(this,
                "Invalid learning rate value. Please enter a positive number.",
                "Invalid Input", JOptionPane.ERROR_MESSAGE);
            return;
        }
        
        try {
            epochs = Integer.parseInt(epochsField.getText().trim());
            if (epochs <= 0) {
                throw new NumberFormatException("Epochs must be positive");
            }
        } catch (NumberFormatException e) {
            JOptionPane.showMessageDialog(this,
                "Invalid epochs value. Please enter a positive integer.",
                "Invalid Input", JOptionPane.ERROR_MESSAGE);
            return;
        }
        
        // Use already converted arrays (more efficient)
        double[][] X = XArray;
        double[][] Y = YArray;
        
        // Create and train network
        statusLabel.setText("Training network... Please wait.");
        trainButton.setEnabled(false);
        
        // Run training in a separate thread to avoid blocking UI
        new Thread(() -> {
            try {
                ann = new ANN(2, hiddenNeurons, NUM_CLASSES, learningRate, 42);
                ann.train(X, Y, epochs, learningRate);
                
                // Update visualization on EDT
                SwingUtilities.invokeLater(() -> {
                    plotPanel.setANN(ann);
                    plotPanel.repaint();
                    statusLabel.setText(String.format("Training complete! Hidden neurons: %d, Learning rate: %.3f, Epochs: %d",
                        hiddenNeurons, learningRate, epochs));
                    trainButton.setEnabled(true);
                });
            } catch (Exception e) {
                SwingUtilities.invokeLater(() -> {
                    JOptionPane.showMessageDialog(this,
                        "Error during training: " + e.getMessage(),
                        "Training Error", JOptionPane.ERROR_MESSAGE);
                    statusLabel.setText("Training failed: " + e.getMessage());
                    trainButton.setEnabled(true);
                });
            }
        }).start();
    }
    
    private void clearData() {
        int result = JOptionPane.showConfirmDialog(this,
            "Are you sure you want to clear all training data?",
            "Clear Data", JOptionPane.YES_NO_OPTION);
        
        if (result == JOptionPane.YES_OPTION) {
            trainingData.clear();
            trainingLabels.clear();
            XArray = new double[0][];
            YArray = new double[0][];
            ann = null;
            plotPanel.setTrainingData(XArray, YArray);
            plotPanel.setANN(null);
            plotPanel.repaint();
            statusLabel.setText("Data cleared. Click on the plot to add training data points.");
        }
    }
    
    /**
     * Interactive boundary panel that allows clicking to add data points
     */
    private class InteractiveBoundaryPanel extends JPanel {
        private final int plotSize = 600;
        private final int padLeft = 60;
        private final int padRight = 20;
        private final int padTop = 20;
        private final int padBottom = 60;
        
        private double[][] X = new double[0][];
        private double[][] Y = new double[0][];
        private ANN ann = null;
        
        // Bounds in input space (normalized to [0, 1] for simplicity)
        private final double minX = 0.0;
        private final double maxX = 1.0;
        private final double minY = 0.0;
        private final double maxY = 1.0;
        
        private BufferedImage background;
        
        private final Color[] regionColors = {
            new Color(255, 120, 120),
            new Color(120, 170, 255),
            new Color(140, 220, 140)
        };
        
        private final Color[] pointColors = {
            new Color(200, 0, 0),
            new Color(0, 70, 200),
            new Color(0, 140, 0)
        };
        
        public InteractiveBoundaryPanel() {
            setPreferredSize(new Dimension(
                padLeft + plotSize + padRight,
                padTop + plotSize + padBottom
            ));
            setBackground(Color.WHITE);
            
            // Add mouse listener for adding points
            addMouseListener(new MouseAdapter() {
                @Override
                public void mousePressed(MouseEvent e) {
                    if (e.getButton() == MouseEvent.BUTTON1) { // Left mouse button only
                        int px = e.getX() - padLeft;
                        int py = e.getY() - padTop;
                        
                        if (px >= 0 && px < plotSize && py >= 0 && py < plotSize) {
                            double x = pxToX(px);
                            double y = pyToY(py);
                            addDataPoint(x, y);
                            e.consume(); // Mark event as handled
                        }
                    }
                }
            });
        }
        
        public void setTrainingData(double[][] X, double[][] Y) {
            this.X = X;
            this.Y = Y;
            // Only repaint the plot area, not the entire component
            repaint(padLeft, padTop, plotSize, plotSize);
        }
        
        public void setANN(ANN ann) {
            this.ann = ann;
            if (ann != null) {
                buildBackground();
            }
            repaint();
        }
        
        private void buildBackground() {
            if (ann == null) return;
            
            background = new BufferedImage(plotSize, plotSize, BufferedImage.TYPE_INT_ARGB);
            
            for (int py = 0; py < plotSize; py++) {
                for (int px = 0; px < plotSize; px++) {
                    double x = pxToX(px);
                    double y = pyToY(py);
                    
                    double[] out = ann.apply(new double[]{x, y});
                    int cls = argMax(out);
                    
                    // Confidence shading
                    double win = out[cls];
                    int alpha = (int) clamp(60 + win * 140, 0, 200);
                    
                    Color base = regionColors[cls % regionColors.length];
                    Color c = new Color(base.getRed(), base.getGreen(), base.getBlue(), alpha);
                    
                    background.setRGB(px, py, c.getRGB());
                }
            }
        }
        
        @Override
        protected void paintComponent(Graphics g0) {
            super.paintComponent(g0);
            Graphics2D g = (Graphics2D) g0.create();
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            
            // Draw background decision regions if trained
            if (background != null) {
                g.drawImage(background, padLeft, padTop, null);
            }
            
            // Frame
            g.setColor(Color.DARK_GRAY);
            g.drawRect(padLeft, padTop, plotSize, plotSize);
            
            // Axes + ticks
            drawAxes(g);
            
            // Draw training points
            drawPoints(g);
            
            // Legend
            drawLegend(g);
            
            g.dispose();
        }
        
        private void drawAxes(Graphics2D g) {
            g.setColor(Color.DARK_GRAY);
            g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 12));
            
            int x0 = padLeft;
            int y0 = padTop + plotSize;
            
            // x-axis label
            g.drawString("x[0]", padLeft + plotSize / 2 - 10, padTop + plotSize + 45);
            // y-axis label
            g.rotate(-Math.PI / 2);
            g.drawString("x[1]", -(padTop + plotSize / 2 + 10), padLeft - 40);
            g.rotate(Math.PI / 2);
            
            int ticks = 5;
            for (int i = 0; i <= ticks; i++) {
                double t = i / (double) ticks;
                
                // X ticks
                int px = padLeft + (int) Math.round(t * plotSize);
                g.drawLine(px, y0, px, y0 + 6);
                double xv = minX + t * (maxX - minX);
                String xs = String.format("%.2f", xv);
                int sw = g.getFontMetrics().stringWidth(xs);
                g.drawString(xs, px - sw / 2, y0 + 20);
                
                // Y ticks
                int py = padTop + plotSize - (int) Math.round(t * plotSize);
                g.drawLine(x0 - 6, py, x0, py);
                double yv = minY + t * (maxY - minY);
                String ys = String.format("%.2f", yv);
                g.drawString(ys, x0 - 10 - g.getFontMetrics().stringWidth(ys), py + g.getFontMetrics().getAscent() / 2 - 2);
            }
        }
        
        private void drawPoints(Graphics2D g) {
            for (int i = 0; i < X.length; i++) {
                double[] p = X[i];
                int cls = argMax(Y[i]);
                
                int px = xToPx(p[0]);
                int py = yToPy(p[1]);
                
                g.setColor(pointColors[cls % pointColors.length]);
                g.fillOval(px - 6, py - 6, 12, 12);
                
                g.setColor(Color.BLACK);
                g.drawOval(px - 6, py - 6, 12, 12);
            }
        }
        
        private void drawLegend(Graphics2D g) {
            int x = padLeft + 10;
            int y = padTop + 10;
            int boxW = 140;
            int boxH = 18 * (NUM_CLASSES + 1);
            
            g.setColor(new Color(255, 255, 255, 220));
            g.fillRoundRect(x, y, boxW, boxH, 10, 10);
            g.setColor(Color.DARK_GRAY);
            g.drawRoundRect(x, y, boxW, boxH, 10, 10);
            
            g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 12));
            g.drawString("Classes", x + 10, y + 15);
            
            g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 12));
            for (int c = 0; c < NUM_CLASSES; c++) {
                int yy = y + 18 + 18 * c;
                g.setColor(pointColors[c % pointColors.length]);
                g.fillOval(x + 10, yy + 2, 10, 10);
                g.setColor(Color.BLACK);
                g.drawOval(x + 10, yy + 2, 10, 10);
                
                g.drawString("Class " + c, x + 30, yy + 12);
            }
        }
        
        // Coordinate transforms
        private double pxToX(int px) {
            double t = px / (double) (plotSize - 1);
            return minX + t * (maxX - minX);
        }
        
        private double pyToY(int py) {
            double t = 1.0 - (py / (double) (plotSize - 1));
            return minY + t * (maxY - minY);
        }
        
        private int xToPx(double x) {
            double t = (x - minX) / (maxX - minX);
            return padLeft + (int) Math.round(t * plotSize);
        }
        
        private int yToPy(double y) {
            double t = (y - minY) / (maxY - minY);
            return padTop + plotSize - (int) Math.round(t * plotSize);
        }
        
        private static int argMax(double[] v) {
            int best = 0;
            for (int i = 1; i < v.length; i++) {
                if (v[i] > v[best]) best = i;
            }
            return best;
        }
        
        private static double clamp(double v, double lo, double hi) {
            return Math.max(lo, Math.min(hi, v));
        }
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            try {
                UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
            } catch (Exception e) {
                // Use default look and feel
            }
            new InteractiveMLPTrainer();
        });
    }
}

