package ch.innunvation;

import ch.innunvation.ann.ANN;

import ch.innunvation.ui.ErrorSurfacePanel;

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
    private ANN initialANN = null; // Store initial weights before training
    
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
                // Store initial weights by creating a copy before training
                initialANN = createANNCopy(ann);
                
                // Enable weight history tracking for all weight pairs we'll visualize
                // This includes various combinations of hidden neurons and inputs
                for (int h1 = 0; h1 < hiddenNeurons && h1 < 3; h1++) {
                    for (int h2 = 0; h2 < hiddenNeurons && h2 < 3; h2++) {
                        // Track same neuron, different inputs
                        if (h1 == h2 && h1 < hiddenNeurons) {
                            ann.enableWeightHistoryTracking(h1, 0, h1, 1);
                        }
                        // Track different neurons, same input
                        if (h1 != h2) {
                            ann.enableWeightHistoryTracking(h1, 0, h2, 0);
                        }
                    }
                }
                // Also track first neuron's weights in case we have fewer than 3 neurons
                if (hiddenNeurons >= 1) {
                    ann.enableWeightHistoryTracking(0, 0, 0, 1);
                }
                
                ann.train(X, Y, epochs, learningRate);
                
                // Update visualization on EDT
                SwingUtilities.invokeLater(() -> {
                    plotPanel.setANN(ann);
                    plotPanel.repaint();
                    statusLabel.setText(String.format("Training complete! Hidden neurons: %d, Learning rate: %.3f, Epochs: %d",
                        hiddenNeurons, learningRate, epochs));
                    trainButton.setEnabled(true);
                    
                    // Show error surface dialog
                    showErrorSurfaceDialog();
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
            initialANN = null;
            plotPanel.setTrainingData(XArray, YArray);
            plotPanel.setANN(null);
            plotPanel.repaint();
            statusLabel.setText("Data cleared. Click on the plot to add training data points.");
        }
    }
    
    private void showErrorSurfaceDialog() {
        if (ann == null || XArray.length == 0) {
            return;
        }
        
        JDialog dialog = new JDialog(this, "Error Surface Visualization - Multiple Weight Configurations", true);
        dialog.setLayout(new BorderLayout());
        
        // Get network dimensions to determine which weights to show
        int nHidden = getHiddenNeurons(ann);
        
        // Create 6 different weight configurations to visualize
        // Using 3x2 grid layout (3 columns, 2 rows) with even spacing
        JPanel gridPanel = new JPanel(new GridLayout(2, 3, 8, 8));
        gridPanel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        gridPanel.setBackground(Color.WHITE);
        
        // Define 6 weight pairs to visualize (most important ones)
        ErrorSurfacePanel[] panels = new ErrorSurfacePanel[6];
        int[][] weightPairs = new int[6][4]; // [hiddenIdx1, inputIdx1, hiddenIdx2, inputIdx2]
        
        // Row 1: Different hidden neurons, both inputs
        // Panel 1: First hidden neuron, both inputs
        weightPairs[0] = new int[]{0, 0, 0, 1};
        panels[0] = new ErrorSurfacePanel(ann, initialANN, XArray, YArray, 
            weightPairs[0][0], weightPairs[0][1], weightPairs[0][2], weightPairs[0][3], 
            "w1[0][0] vs w1[0][1]", ann.getWeightHistory(weightPairs[0][0], weightPairs[0][1], weightPairs[0][2], weightPairs[0][3]));
        
        // Panel 2: Second hidden neuron, both inputs
        if (nHidden > 1) {
            weightPairs[1] = new int[]{1, 0, 1, 1};
            panels[1] = new ErrorSurfacePanel(ann, initialANN, XArray, YArray, 
                weightPairs[1][0], weightPairs[1][1], weightPairs[1][2], weightPairs[1][3], 
                "w1[1][0] vs w1[1][1]", ann.getWeightHistory(weightPairs[1][0], weightPairs[1][1], weightPairs[1][2], weightPairs[1][3]));
        } else {
            weightPairs[1] = new int[]{0, 0, 0, 1};
            panels[1] = new ErrorSurfacePanel(ann, initialANN, XArray, YArray, 
                weightPairs[1][0], weightPairs[1][1], weightPairs[1][2], weightPairs[1][3], 
                "w1[0][0] vs w1[0][1]", ann.getWeightHistory(weightPairs[1][0], weightPairs[1][1], weightPairs[1][2], weightPairs[1][3]));
        }
        
        // Panel 3: Third hidden neuron, both inputs
        if (nHidden > 2) {
            weightPairs[2] = new int[]{2, 0, 2, 1};
            panels[2] = new ErrorSurfacePanel(ann, initialANN, XArray, YArray, 
                weightPairs[2][0], weightPairs[2][1], weightPairs[2][2], weightPairs[2][3], 
                "w1[2][0] vs w1[2][1]", ann.getWeightHistory(weightPairs[2][0], weightPairs[2][1], weightPairs[2][2], weightPairs[2][3]));
        } else {
            weightPairs[2] = new int[]{0, 0, 0, 1};
            panels[2] = new ErrorSurfacePanel(ann, initialANN, XArray, YArray, 
                weightPairs[2][0], weightPairs[2][1], weightPairs[2][2], weightPairs[2][3], 
                "w1[0][0] vs w1[0][1]", ann.getWeightHistory(weightPairs[2][0], weightPairs[2][1], weightPairs[2][2], weightPairs[2][3]));
        }
        
        // Row 2: Cross-neuron comparisons, input 0
        // Panel 4: Hidden neurons 0 and 1, input 0
        if (nHidden > 1) {
            weightPairs[3] = new int[]{0, 0, 1, 0};
            panels[3] = new ErrorSurfacePanel(ann, initialANN, XArray, YArray, 
                weightPairs[3][0], weightPairs[3][1], weightPairs[3][2], weightPairs[3][3], 
                "w1[0][0] vs w1[1][0]", ann.getWeightHistory(weightPairs[3][0], weightPairs[3][1], weightPairs[3][2], weightPairs[3][3]));
        } else {
            weightPairs[3] = new int[]{0, 0, 0, 1};
            panels[3] = new ErrorSurfacePanel(ann, initialANN, XArray, YArray, 
                weightPairs[3][0], weightPairs[3][1], weightPairs[3][2], weightPairs[3][3], 
                "w1[0][0] vs w1[0][1]", ann.getWeightHistory(weightPairs[3][0], weightPairs[3][1], weightPairs[3][2], weightPairs[3][3]));
        }
        
        // Panel 5: Hidden neurons 1 and 2, input 0
        if (nHidden > 2) {
            weightPairs[4] = new int[]{1, 0, 2, 0};
            panels[4] = new ErrorSurfacePanel(ann, initialANN, XArray, YArray, 
                weightPairs[4][0], weightPairs[4][1], weightPairs[4][2], weightPairs[4][3], 
                "w1[1][0] vs w1[2][0]", ann.getWeightHistory(weightPairs[4][0], weightPairs[4][1], weightPairs[4][2], weightPairs[4][3]));
        } else if (nHidden > 1) {
            weightPairs[4] = new int[]{0, 0, 1, 0};
            panels[4] = new ErrorSurfacePanel(ann, initialANN, XArray, YArray, 
                weightPairs[4][0], weightPairs[4][1], weightPairs[4][2], weightPairs[4][3], 
                "w1[0][0] vs w1[1][0]", ann.getWeightHistory(weightPairs[4][0], weightPairs[4][1], weightPairs[4][2], weightPairs[4][3]));
        } else {
            weightPairs[4] = new int[]{0, 0, 0, 1};
            panels[4] = new ErrorSurfacePanel(ann, initialANN, XArray, YArray, 
                weightPairs[4][0], weightPairs[4][1], weightPairs[4][2], weightPairs[4][3], 
                "w1[0][0] vs w1[0][1]", ann.getWeightHistory(weightPairs[4][0], weightPairs[4][1], weightPairs[4][2], weightPairs[4][3]));
        }
        
        // Panel 6: Hidden neurons 0 and 2, input 0
        if (nHidden > 2) {
            weightPairs[5] = new int[]{0, 0, 2, 0};
            panels[5] = new ErrorSurfacePanel(ann, initialANN, XArray, YArray, 
                weightPairs[5][0], weightPairs[5][1], weightPairs[5][2], weightPairs[5][3], 
                "w1[0][0] vs w1[2][0]", ann.getWeightHistory(weightPairs[5][0], weightPairs[5][1], weightPairs[5][2], weightPairs[5][3]));
        } else {
            weightPairs[5] = new int[]{0, 0, 0, 1};
            panels[5] = new ErrorSurfacePanel(ann, initialANN, XArray, YArray, 
                weightPairs[5][0], weightPairs[5][1], weightPairs[5][2], weightPairs[5][3], 
                "w1[0][0] vs w1[0][1]", ann.getWeightHistory(weightPairs[5][0], weightPairs[5][1], weightPairs[5][2], weightPairs[5][3]));
        }
        
        // Add all panels to grid with centered borders
        for (ErrorSurfacePanel panel : panels) {
            JPanel container = new JPanel(new BorderLayout());
            container.setBorder(BorderFactory.createTitledBorder(
                BorderFactory.createEtchedBorder(),
                panel.getWeightDescription(),
                javax.swing.border.TitledBorder.CENTER,
                javax.swing.border.TitledBorder.TOP,
                new Font(Font.SANS_SERIF, Font.BOLD, 10) // Smaller font
            ));
            container.setBackground(Color.WHITE);
            container.setOpaque(true);
            // Add panel directly - don't wrap in another panel
            container.add(panel, BorderLayout.CENTER);
            gridPanel.add(container);
            
            // Force panel to be visible and validate
            panel.setVisible(true);
            panel.setOpaque(true);
        }
        
        // Add grid directly (no scroll pane) so it fits fullscreen
        dialog.add(gridPanel, BorderLayout.CENTER);
        
        // Add info panel (compact)
        JPanel infoPanel = new JPanel();
        infoPanel.setLayout(new BoxLayout(infoPanel, BoxLayout.Y_AXIS));
        infoPanel.setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createTitledBorder("Visualization Info"),
            BorderFactory.createEmptyBorder(3, 8, 3, 8)
        ));
        infoPanel.setBackground(Color.WHITE);
        
        int totalParams = getTotalParameters(ann);
        String infoText = String.format(
            "6 different 2D slices of the error surface. Total parameters: %d. Drag to rotate, scroll to zoom.",
            totalParams
        );
        
        JLabel infoLabel = new JLabel(infoText);
        infoLabel.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 10));
        infoPanel.add(infoLabel);
        
        dialog.add(infoPanel, BorderLayout.NORTH);
        
        // Add close button (compact)
        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));
        buttonPanel.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
        JButton closeButton = new JButton("Close");
        closeButton.setPreferredSize(new Dimension(80, 25));
        closeButton.addActionListener(e -> dialog.dispose());
        buttonPanel.add(closeButton);
        dialog.add(buttonPanel, BorderLayout.SOUTH);
        
        // Set dialog size to fit fullscreen (slightly smaller than screen)
        Toolkit toolkit = Toolkit.getDefaultToolkit();
        Dimension screenSize = toolkit.getScreenSize();
        dialog.setSize((int)(screenSize.width * 0.95), (int)(screenSize.height * 0.95));
        dialog.setLocationRelativeTo(this);
        
        // Validate and pack to ensure components are laid out
        dialog.validate();
        dialog.setVisible(true);
    }
    
    private int getHiddenNeurons(ANN ann) {
        try {
            java.lang.reflect.Field field = ANN.class.getDeclaredField("nHidden");
            field.setAccessible(true);
            return field.getInt(ann);
        } catch (Exception e) {
            return 6; // Default
        }
    }
    
    private int getTotalParameters(ANN ann) {
        try {
            java.lang.reflect.Field nInField = ANN.class.getDeclaredField("nIn");
            java.lang.reflect.Field nHiddenField = ANN.class.getDeclaredField("nHidden");
            java.lang.reflect.Field nOutField = ANN.class.getDeclaredField("nOut");
            nInField.setAccessible(true);
            nHiddenField.setAccessible(true);
            nOutField.setAccessible(true);
            
            int nIn = nInField.getInt(ann);
            int nHidden = nHiddenField.getInt(ann);
            int nOut = nOutField.getInt(ann);
            
            return (nIn * nHidden) + nHidden + (nHidden * nOut) + nOut;
        } catch (Exception e) {
            return 39; // Default estimate
        }
    }
    
    /**
     * Creates a copy of an ANN with the same initial weights
     */
    private ANN createANNCopy(ANN original) {
        try {
            java.lang.reflect.Field nInField = ANN.class.getDeclaredField("nIn");
            java.lang.reflect.Field nHiddenField = ANN.class.getDeclaredField("nHidden");
            java.lang.reflect.Field nOutField = ANN.class.getDeclaredField("nOut");
            java.lang.reflect.Field w1Field = ANN.class.getDeclaredField("w1");
            java.lang.reflect.Field b1Field = ANN.class.getDeclaredField("b1");
            java.lang.reflect.Field w2Field = ANN.class.getDeclaredField("w2");
            java.lang.reflect.Field b2Field = ANN.class.getDeclaredField("b2");
            java.lang.reflect.Field defaultLRField = ANN.class.getDeclaredField("defaultLearningRate");
            
            nInField.setAccessible(true);
            nHiddenField.setAccessible(true);
            nOutField.setAccessible(true);
            w1Field.setAccessible(true);
            b1Field.setAccessible(true);
            w2Field.setAccessible(true);
            b2Field.setAccessible(true);
            defaultLRField.setAccessible(true);
            
            int nIn = nInField.getInt(original);
            int nHidden = nHiddenField.getInt(original);
            int nOut = nOutField.getInt(original);
            double defaultLR = defaultLRField.getDouble(original);
            
            // Create new ANN with same seed to get same initial weights
            // Actually, we need to copy the weights manually since we can't control the seed
            ANN copy = new ANN(nIn, nHidden, nOut, defaultLR, new java.util.Random());
            
            // Copy weights and biases
            double[][] w1 = (double[][]) w1Field.get(original);
            double[] b1 = (double[]) b1Field.get(original);
            double[][] w2 = (double[][]) w2Field.get(original);
            double[] b2 = (double[]) b2Field.get(original);
            
            double[][] copyW1 = (double[][]) w1Field.get(copy);
            double[] copyB1 = (double[]) b1Field.get(copy);
            double[][] copyW2 = (double[][]) w2Field.get(copy);
            double[] copyB2 = (double[]) b2Field.get(copy);
            
            // Deep copy weights
            for (int i = 0; i < w1.length; i++) {
                System.arraycopy(w1[i], 0, copyW1[i], 0, w1[i].length);
            }
            System.arraycopy(b1, 0, copyB1, 0, b1.length);
            for (int i = 0; i < w2.length; i++) {
                System.arraycopy(w2[i], 0, copyW2[i], 0, w2[i].length);
            }
            System.arraycopy(b2, 0, copyB2, 0, b2.length);
            
            return copy;
        } catch (Exception e) {
            System.err.println("Error creating ANN copy: " + e.getMessage());
            e.printStackTrace();
            return null;
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
            } else {
                // Clear background when ANN is null (e.g., when data is cleared)
                background = null;
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

