package ch.innunvation;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

public class BoundaryPanelBinary extends JPanel {

    private final ANN ann;
    private final double[][] X;     // training inputs (Nx2)
    private final int[] yClass;     // class label 0/1 for each sample

    // World bounds (data space)
    private final double minX, maxX, minY, maxY;

    // Rendering config
    private final int pad = 55;       // space for axes + labels
    private final int gridW = 500;    // resolution of background
    private final int gridH = 500;

    private BufferedImage background; // cached background

    public BoundaryPanelBinary(ANN ann, double[][] X, int[] yClass,
                               double minX, double maxX, double minY, double maxY) {
        this.ann = ann;
        this.X = X;
        this.yClass = yClass;
        this.minX = minX;
        this.maxX = maxX;
        this.minY = minY;
        this.maxY = maxY;

        setPreferredSize(new Dimension(700, 700));
        setBackground(Color.WHITE);
    }

    /** Rebuild background after training (call once after ann.train(...)). */
    public void rebuildBackground() {
        background = new BufferedImage(gridW, gridH, BufferedImage.TYPE_INT_ARGB);

        // Two class colors (like the matplotlib example: bluish vs reddish)
        Color c0 = new Color(240, 120, 120); // class 0-ish
        Color c1 = new Color(120, 160, 240); // class 1-ish

        for (int iy = 0; iy < gridH; iy++) {
            double y = lerp(maxY, minY, iy / (double)(gridH - 1)); // top->bottom
            for (int ix = 0; ix < gridW; ix++) {
                double x = lerp(minX, maxX, ix / (double)(gridW - 1));

                double p = ann.apply(new double[]{x, y})[0]; // probability of class 1
                // blend colors based on p
                Color blended = blend(c0, c1, p);

                // light alpha so points/boundary stand out
                int argb = (180 << 24) | (blended.getRed() << 16) | (blended.getGreen() << 8) | blended.getBlue();
                background.setRGB(ix, iy, argb);
            }
        }
        repaint();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g.create();
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        int w = getWidth();
        int h = getHeight();

        // Plot area
        int px0 = pad;
        int py0 = pad;
        int pw = w - 2 * pad;
        int ph = h - 2 * pad;

        // Draw background (decision surface)
        if (background != null) {
            g2.drawImage(background, px0, py0, pw, ph, null);
        } else {
            // If not built yet, show a hint
            g2.setColor(Color.LIGHT_GRAY);
            g2.drawRect(px0, py0, pw, ph);
            g2.drawString("Call panel.rebuildBackground() after training", px0 + 10, py0 + 20);
        }

        // Draw decision boundary p ~= 0.5 as small dots (simple & clean)
        if (background != null) {
            g2.setColor(new Color(255, 255, 255, 220));
            int step = 2;              // boundary sampling
            double eps = 0.02;         // how close to 0.5 counts as boundary
            for (int sy = 0; sy < ph; sy += step) {
                double y = screenToWorldY(py0 + sy, py0, ph);
                for (int sx = 0; sx < pw; sx += step) {
                    double x = screenToWorldX(px0 + sx, px0, pw);
                    double p = ann.apply(new double[]{x, y})[0];
                    if (Math.abs(p - 0.5) < eps) {
                        g2.fillRect(px0 + sx, py0 + sy, 1, 1);
                    }
                }
            }
        }

        // Draw training points
        for (int i = 0; i < X.length; i++) {
            int sx = worldToScreenX(X[i][0], px0, pw);
            int sy = worldToScreenY(X[i][1], py0, ph);

            if (yClass[i] == 0) {
                g2.setColor(new Color(200, 30, 30));
            } else {
                g2.setColor(new Color(30, 90, 200));
            }

            int r = 6;
            g2.fillOval(sx - r, sy - r, 2 * r, 2 * r);
            g2.setColor(Color.BLACK);
            g2.drawOval(sx - r, sy - r, 2 * r, 2 * r);
        }

        // Axes + ticks + labels
        drawAxes(g2, px0, py0, pw, ph);

        g2.dispose();
    }

    private void drawAxes(Graphics2D g2, int px0, int py0, int pw, int ph) {
        g2.setColor(Color.BLACK);
        g2.setStroke(new BasicStroke(1.2f));

        // Border of plot area
        g2.drawRect(px0, py0, pw, ph);

        // Ticks
        int ticks = 6;
        FontMetrics fm = g2.getFontMetrics();

        // X ticks
        for (int i = 0; i <= ticks; i++) {
            double t = i / (double) ticks;
            double xVal = lerp(minX, maxX, t);
            int x = px0 + (int) Math.round(t * pw);

            g2.drawLine(x, py0 + ph, x, py0 + ph + 6);
            String s = String.format("%.2f", xVal);
            int sw = fm.stringWidth(s);
            g2.drawString(s, x - sw / 2, py0 + ph + 20);
        }

        // Y ticks
        for (int i = 0; i <= ticks; i++) {
            double t = i / (double) ticks;
            double yVal = lerp(maxY, minY, t); // top is maxY
            int y = py0 + (int) Math.round(t * ph);

            g2.drawLine(px0 - 6, y, px0, y);
            String s = String.format("%.2f", yVal);
            int sw = fm.stringWidth(s);
            g2.drawString(s, px0 - 10 - sw, y + fm.getAscent() / 2 - 2);
        }

        // Axis labels
        g2.drawString("x0", px0 + pw / 2 - 8, py0 + ph + 45);

        // rotated y label
        Graphics2D gR = (Graphics2D) g2.create();
        gR.rotate(-Math.PI / 2);
        gR.drawString("x1", -(py0 + ph / 2 + 8), px0 - 40);
        gR.dispose();
    }

    // --- coordinate mapping helpers ---
    private int worldToScreenX(double x, int px0, int pw) {
        double t = (x - minX) / (maxX - minX);
        return px0 + (int) Math.round(t * pw);
    }

    private int worldToScreenY(double y, int py0, int ph) {
        double t = (maxY - y) / (maxY - minY); // invert
        return py0 + (int) Math.round(t * ph);
    }

    private double screenToWorldX(int sx, int px0, int pw) {
        double t = (sx - px0) / (double) pw;
        return lerp(minX, maxX, t);
    }

    private double screenToWorldY(int sy, int py0, int ph) {
        double t = (sy - py0) / (double) ph;
        return lerp(maxY, minY, t);
    }

    private static double lerp(double a, double b, double t) {
        return a + (b - a) * t;
    }

    private static Color blend(Color a, Color b, double t) {
        t = Math.max(0, Math.min(1, t));
        int r = (int) Math.round(a.getRed()   * (1 - t) + b.getRed()   * t);
        int g = (int) Math.round(a.getGreen() * (1 - t) + b.getGreen() * t);
        int bl= (int) Math.round(a.getBlue()  * (1 - t) + b.getBlue()  * t);
        return new Color(r, g, bl);
    }
}
