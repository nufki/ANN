package ch.innunvation;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

public class BoundaryPanelMulti extends JPanel {

    private final ANN ann;
    private final double[][] X;
    private final double[][] Y;
    private final int nClasses;

    // Plot config
    private final int plotSize = 600;
    private final int padLeft = 60;
    private final int padRight = 20;
    private final int padTop = 20;
    private final int padBottom = 60;

    private BufferedImage background;

    // bounds in input space
    private double minX, maxX, minY, maxY;

    // Simple palette (background regions + point colors)
    private final Color[] regionColors;
    private final Color[] pointColors;

    public BoundaryPanelMulti(ANN ann, double[][] X, double[][] Y, int nClasses) {
        this.ann = ann;
        this.X = X;
        this.Y = Y;
        this.nClasses = nClasses;

        this.regionColors = makeRegionPalette(nClasses);
        this.pointColors = makePointPalette(nClasses);

        computeBounds();
        buildBackground();

        int w = padLeft + plotSize + padRight;
        int h = padTop + plotSize + padBottom;
        setPreferredSize(new Dimension(w, h));
        setBackground(Color.WHITE);
    }

    private void computeBounds() {
        minX = Double.POSITIVE_INFINITY;
        maxX = Double.NEGATIVE_INFINITY;
        minY = Double.POSITIVE_INFINITY;
        maxY = Double.NEGATIVE_INFINITY;

        for (double[] p : X) {
            minX = Math.min(minX, p[0]);
            maxX = Math.max(maxX, p[0]);
            minY = Math.min(minY, p[1]);
            maxY = Math.max(maxY, p[1]);
        }

        // add margin
        double mx = (maxX - minX) * 0.30 + 1e-9;
        double my = (maxY - minY) * 0.30 + 1e-9;
        minX -= mx; maxX += mx;
        minY -= my; maxY += my;
    }

    private void buildBackground() {
        background = new BufferedImage(plotSize, plotSize, BufferedImage.TYPE_INT_ARGB);

        for (int py = 0; py < plotSize; py++) {
            for (int px = 0; px < plotSize; px++) {

                double x = pxToX(px);
                double y = pyToY(py);

                double[] out = ann.apply(new double[]{x, y});
                int cls = argMax(out);

                // confidence shading: use winning score (0..1) to adjust alpha a bit
                double win = out[cls];
                int alpha = (int) clamp(60 + win * 140, 0, 200); // soft pastel regions

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

        // Draw background decision regions
        g.drawImage(background, padLeft, padTop, null);

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
            int sh = g.getFontMetrics().getAscent();
            g.drawString(ys, x0 - 10 - g.getFontMetrics().stringWidth(ys), py + sh / 2 - 2);
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
        int boxH = 18 * (nClasses + 1);

        g.setColor(new Color(255, 255, 255, 220));
        g.fillRoundRect(x, y, boxW, boxH, 10, 10);
        g.setColor(Color.DARK_GRAY);
        g.drawRoundRect(x, y, boxW, boxH, 10, 10);

        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 12));
        g.drawString("Classes", x + 10, y + 15);

        g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 12));
        for (int c = 0; c < nClasses; c++) {
            int yy = y + 18 + 18 * c;
            g.setColor(pointColors[c % pointColors.length]);
            g.fillOval(x + 10, yy + 2, 10, 10);
            g.setColor(Color.BLACK);
            g.drawOval(x + 10, yy + 2, 10, 10);

            g.drawString("Class " + c, x + 30, yy + 12);
        }
    }

    // --- Coordinate transforms ---
    private double pxToX(int px) {
        double t = px / (double) (plotSize - 1);
        return minX + t * (maxX - minX);
    }

    private double pyToY(int py) {
        // py=0 is top -> maxY, py=plotSize-1 is bottom -> minY
        double t = 1.0 - (py / (double) (plotSize - 1));
        return minY + t * (maxY - minY);
    }

    private int xToPx(double x) {
        double t = (x - minX) / (maxX - minX);
        return padLeft + (int) Math.round(t * plotSize);
    }

    private int yToPy(double y) {
        double t = (y - minY) / (maxY - minY);
        // invert
        return padTop + plotSize - (int) Math.round(t * plotSize);
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
