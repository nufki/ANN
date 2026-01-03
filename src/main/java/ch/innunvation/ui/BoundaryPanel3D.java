package ch.innunvation.ui;

import ch.innunvation.ann.ANN;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

public class BoundaryPanel3D extends JPanel {

    private final ANN ann;
    private final double[][] X;
    private final double[][] Y;
    private final int nClasses;

    // Plot config
    private final int plotSize = 400;
    private final int padLeft = 60;
    private final int padRight = 20;
    private final int padTop = 60;
    private final int padBottom = 20;

    private BufferedImage backgroundXY, backgroundXZ, backgroundYZ;

    // bounds in input space
    private double minX, maxX, minY, maxY, minZ, maxZ;

    // Current slice positions
    private double sliceZ = 0.5; // for XY plane
    private double sliceY = 0.5; // for XZ plane
    private double sliceX = 0.5; // for YZ plane

    // Simple palette (background regions + point colors)
    private final Color[] regionColors;
    private final Color[] pointColors;

    // Sliders for interactive control
    private JSlider sliderZ, sliderY, sliderX;

    public BoundaryPanel3D(ANN ann, double[][] X, double[][] Y, int nClasses) {
        this.ann = ann;
        this.X = X;
        this.Y = Y;
        this.nClasses = nClasses;

        this.regionColors = makeRegionPalette(nClasses);
        this.pointColors = makePointPalette(nClasses);

        computeBounds();
        buildUI();
        rebuildBackgrounds();
    }

    private void computeBounds() {
        minX = Double.POSITIVE_INFINITY;
        maxX = Double.NEGATIVE_INFINITY;
        minY = Double.POSITIVE_INFINITY;
        maxY = Double.NEGATIVE_INFINITY;
        minZ = Double.POSITIVE_INFINITY;
        maxZ = Double.NEGATIVE_INFINITY;

        for (double[] p : X) {
            minX = Math.min(minX, p[0]);
            maxX = Math.max(maxX, p[0]);
            minY = Math.min(minY, p[1]);
            maxY = Math.max(maxY, p[1]);
            minZ = Math.min(minZ, p[2]);
            maxZ = Math.max(maxZ, p[2]);
        }

        // add margin
        double mx = (maxX - minX) * 0.30 + 1e-9;
        double my = (maxY - minY) * 0.30 + 1e-9;
        double mz = (maxZ - minZ) * 0.30 + 1e-9;
        minX -= mx; maxX += mx;
        minY -= my; maxY += my;
        minZ -= mz; maxZ += mz;

        // Initialize slice positions to middle
        sliceZ = (minZ + maxZ) / 2.0;
        sliceY = (minY + maxY) / 2.0;
        sliceX = (minX + maxX) / 2.0;
    }

    private void buildUI() {
        setLayout(new BorderLayout());

        // Main visualization panel that draws all three slices
        JPanel mainPanel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g0) {
                super.paintComponent(g0);
                Graphics2D g = (Graphics2D) g0.create();
                g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

                // Draw XY plane (top-left)
                drawSlice(g, backgroundXY, 0, 0, "XY Plane (Z = " + String.format("%.2f", sliceZ) + ")");
                // Draw XZ plane (top-right)
                drawSlice(g, backgroundXZ, 1, 0, "XZ Plane (Y = " + String.format("%.2f", sliceY) + ")");
                // Draw YZ plane (bottom-left)
                drawSlice(g, backgroundYZ, 0, 1, "YZ Plane (X = " + String.format("%.2f", sliceX) + ")");

                g.dispose();
            }
        };
        mainPanel.setPreferredSize(new Dimension(
                padLeft + plotSize + padRight + padLeft + plotSize + padRight + 20,
                padTop + plotSize + padBottom + padTop + plotSize + padBottom + 20));
        mainPanel.setBackground(Color.WHITE);

        // Control panel with sliders
        JPanel controlPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 20, 10));

        // Z slider for XY plane
        JPanel panelZ = new JPanel(new BorderLayout());
        panelZ.add(new JLabel("Z slice:"), BorderLayout.NORTH);
        sliderZ = new JSlider(0, 100, 50);
        sliderZ.addChangeListener(e -> {
            sliceZ = minZ + (sliderZ.getValue() / 100.0) * (maxZ - minZ);
            rebuildXY();
            repaint();
        });
        panelZ.add(sliderZ, BorderLayout.CENTER);

        // Y slider for XZ plane
        JPanel panelY = new JPanel(new BorderLayout());
        panelY.add(new JLabel("Y slice:"), BorderLayout.NORTH);
        sliderY = new JSlider(0, 100, 50);
        sliderY.addChangeListener(e -> {
            sliceY = minY + (sliderY.getValue() / 100.0) * (maxY - minY);
            rebuildXZ();
            repaint();
        });
        panelY.add(sliderY, BorderLayout.CENTER);

        // X slider for YZ plane
        JPanel panelX = new JPanel(new BorderLayout());
        panelX.add(new JLabel("X slice:"), BorderLayout.NORTH);
        sliderX = new JSlider(0, 100, 50);
        sliderX.addChangeListener(e -> {
            sliceX = minX + (sliderX.getValue() / 100.0) * (maxX - minX);
            rebuildYZ();
            repaint();
        });
        panelX.add(sliderX, BorderLayout.CENTER);

        controlPanel.add(panelZ);
        controlPanel.add(panelY);
        controlPanel.add(panelX);

        // Legend panel
        JPanel legendPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 20, 5));
        legendPanel.setBackground(Color.WHITE);
        legendPanel.add(new JLabel("Classes:"));
        for (int c = 0; c < nClasses; c++) {
            JPanel classPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 5, 0));
            JLabel colorLabel = new JLabel("●");
            colorLabel.setForeground(pointColors[c % pointColors.length]);
            colorLabel.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 20));
            classPanel.add(colorLabel);
            classPanel.add(new JLabel("Class " + c));
            classPanel.setBackground(Color.WHITE);
            legendPanel.add(classPanel);
        }

        add(mainPanel, BorderLayout.CENTER);
        add(controlPanel, BorderLayout.SOUTH);
        add(legendPanel, BorderLayout.NORTH);

        setBackground(Color.WHITE);
    }

    private void drawSlice(Graphics2D g, BufferedImage bg, int col, int row, String title) {
        int spacing = 10;
        int x = spacing + (padLeft + plotSize + padRight) * col;
        int y = spacing + (padTop + plotSize + padBottom) * row;

        // Draw background decision regions
        if (bg != null) {
            g.drawImage(bg, x, y, null);
        }

        // Frame
        g.setColor(Color.DARK_GRAY);
        g.drawRect(x, y, plotSize, plotSize);

        // Title
        g.setColor(Color.BLACK);
        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 12));
        FontMetrics fm = g.getFontMetrics();
        int titleWidth = fm.stringWidth(title);
        g.drawString(title, x + (plotSize - titleWidth) / 2, y - 10);

        // Axes + ticks
        drawAxes(g, x, y, col, row);

        // Draw training points visible in this slice
        drawPointsInSlice(g, x, y, col, row);
    }

    private void drawAxes(Graphics2D g, int x0, int y0, int col, int row) {
        g.setColor(Color.DARK_GRAY);
        g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 10));

        String xLabel, yLabel;
        double xMin, xMax, yMin, yMax;
        
        if (col == 0 && row == 0) {
            // XY plane
            xLabel = "X";
            yLabel = "Y";
            xMin = minX; xMax = maxX;
            yMin = minY; yMax = maxY;
        } else if (col == 1 && row == 0) {
            // XZ plane
            xLabel = "X";
            yLabel = "Z";
            xMin = minX; xMax = maxX;
            yMin = minZ; yMax = maxZ;
        } else {
            // YZ plane
            xLabel = "Y";
            yLabel = "Z";
            xMin = minY; xMax = maxY;
            yMin = minZ; yMax = maxZ;
        }

        // X-axis label
        int xAxisY = y0 + plotSize + 15;
        g.drawString(xLabel, x0 + plotSize / 2 - 5, xAxisY);

        // Y-axis label
        g.rotate(-Math.PI / 2);
        g.drawString(yLabel, -(y0 + plotSize / 2 + 5), x0 - 40);
        g.rotate(Math.PI / 2);

        int ticks = 4;
        for (int i = 0; i <= ticks; i++) {
            double t = i / (double) ticks;

            // X ticks
            int px = x0 + (int) Math.round(t * plotSize);
            g.drawLine(px, y0 + plotSize, px, y0 + plotSize + 4);
            double xv = xMin + t * (xMax - xMin);
            String xs = String.format("%.1f", xv);
            int sw = g.getFontMetrics().stringWidth(xs);
            g.drawString(xs, px - sw / 2, xAxisY + 12);

            // Y ticks
            int py = y0 + plotSize - (int) Math.round(t * plotSize);
            g.drawLine(x0 - 4, py, x0, py);
            double yv = yMin + t * (yMax - yMin);
            String ys = String.format("%.1f", yv);
            int sw2 = g.getFontMetrics().stringWidth(ys);
            g.drawString(ys, x0 - 8 - sw2, py + 4);
        }
    }

    private void drawPointsInSlice(Graphics2D g, int x0, int y0, int col, int row) {
        double threshold = 0.05; // point is visible if within threshold of slice
        
        for (int i = 0; i < X.length; i++) {
            double[] p = X[i];
            boolean visible = false;
            double px = 0, py = 0;
            
            if (col == 0 && row == 0) {
                // XY plane - show if Z is close to sliceZ
                if (Math.abs(p[2] - sliceZ) < threshold * (maxZ - minZ)) {
                    visible = true;
                    px = p[0];
                    py = p[1];
                }
            } else if (col == 1 && row == 0) {
                // XZ plane - show if Y is close to sliceY
                if (Math.abs(p[1] - sliceY) < threshold * (maxY - minY)) {
                    visible = true;
                    px = p[0];
                    py = p[2];
                }
            } else {
                // YZ plane - show if X is close to sliceX
                if (Math.abs(p[0] - sliceX) < threshold * (maxX - minX)) {
                    visible = true;
                    px = p[1];
                    py = p[2];
                }
            }
            
            if (visible) {
                int cls = argMax(Y[i]);
                
                double xMin, xMax, yMin, yMax;
                if (col == 0 && row == 0) {
                    xMin = minX; xMax = maxX; yMin = minY; yMax = maxY;
                } else if (col == 1 && row == 0) {
                    xMin = minX; xMax = maxX; yMin = minZ; yMax = maxZ;
                } else {
                    xMin = minY; xMax = maxY; yMin = minZ; yMax = maxZ;
                }
                
                int sx = x0 + (int) Math.round(((px - xMin) / (xMax - xMin)) * plotSize);
                int sy = y0 + plotSize - (int) Math.round(((py - yMin) / (yMax - yMin)) * plotSize);
                
                g.setColor(pointColors[cls % pointColors.length]);
                g.fillOval(sx - 5, sy - 5, 10, 10);
                g.setColor(Color.BLACK);
                g.drawOval(sx - 5, sy - 5, 10, 10);
            }
        }
    }

    private void rebuildBackgrounds() {
        rebuildXY();
        rebuildXZ();
        rebuildYZ();
    }

    private void rebuildXY() {
        backgroundXY = new BufferedImage(plotSize, plotSize, BufferedImage.TYPE_INT_ARGB);
        for (int py = 0; py < plotSize; py++) {
            for (int px = 0; px < plotSize; px++) {
                double x = pxToX(px, minX, maxX);
                double y = pyToY(py, minY, maxY);
                double[] out = ann.apply(new double[]{x, y, sliceZ});
                int cls = argMax(out);
                double win = out[cls];
                int alpha = (int) clamp(60 + win * 140, 0, 200);
                Color base = regionColors[cls % regionColors.length];
                Color c = new Color(base.getRed(), base.getGreen(), base.getBlue(), alpha);
                backgroundXY.setRGB(px, py, c.getRGB());
            }
        }
    }

    private void rebuildXZ() {
        backgroundXZ = new BufferedImage(plotSize, plotSize, BufferedImage.TYPE_INT_ARGB);
        for (int py = 0; py < plotSize; py++) {
            for (int px = 0; px < plotSize; px++) {
                double x = pxToX(px, minX, maxX);
                double z = pyToY(py, minZ, maxZ);
                double[] out = ann.apply(new double[]{x, sliceY, z});
                int cls = argMax(out);
                double win = out[cls];
                int alpha = (int) clamp(60 + win * 140, 0, 200);
                Color base = regionColors[cls % regionColors.length];
                Color c = new Color(base.getRed(), base.getGreen(), base.getBlue(), alpha);
                backgroundXZ.setRGB(px, py, c.getRGB());
            }
        }
    }

    private void rebuildYZ() {
        backgroundYZ = new BufferedImage(plotSize, plotSize, BufferedImage.TYPE_INT_ARGB);
        for (int py = 0; py < plotSize; py++) {
            for (int px = 0; px < plotSize; px++) {
                double y = pxToX(px, minY, maxY);
                double z = pyToY(py, minZ, maxZ);
                double[] out = ann.apply(new double[]{sliceX, y, z});
                int cls = argMax(out);
                double win = out[cls];
                int alpha = (int) clamp(60 + win * 140, 0, 200);
                Color base = regionColors[cls % regionColors.length];
                Color c = new Color(base.getRed(), base.getGreen(), base.getBlue(), alpha);
                backgroundYZ.setRGB(px, py, c.getRGB());
            }
        }
    }

    // --- Coordinate transforms ---
    private double pxToX(int px, double min, double max) {
        double t = px / (double) (plotSize - 1);
        return min + t * (max - min);
    }

    private double pyToY(int py, double min, double max) {
        double t = 1.0 - (py / (double) (plotSize - 1));
        return min + t * (max - min);
    }

    // --- utils ---
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

    private static Color[] makeRegionPalette(int k) {
        // pastel-ish backgrounds
        Color[] base = {
                new Color(255, 120, 120),
                new Color(120, 170, 255),
                new Color(140, 220, 140),
                new Color(255, 200, 120),
                new Color(200, 140, 255),
        };
        Color[] out = new Color[k];
        for (int i = 0; i < k; i++) out[i] = base[i % base.length];
        return out;
    }

    private static Color[] makePointPalette(int k) {
        // stronger point colors
        Color[] base = {
                new Color(200, 0, 0),
                new Color(0, 70, 200),
                new Color(0, 140, 0),
                new Color(200, 120, 0),
                new Color(120, 0, 200),
        };
        Color[] out = new Color[k];
        for (int i = 0; i < k; i++) out[i] = base[i % base.length];
        return out;
    }
}

