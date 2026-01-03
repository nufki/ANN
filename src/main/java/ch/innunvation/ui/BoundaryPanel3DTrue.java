package ch.innunvation.ui;

import ch.innunvation.ann.ANN;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.image.BufferedImage;

/**
 * True 3D visualization panel with interactive rotation.
 * Uses 3D projection mathematics (rotation matrices + perspective/orthographic projection)
 * to render 3D points and decision boundaries without requiring external 3D libraries.
 */
public class BoundaryPanel3DTrue extends JPanel {

    private final ANN ann;
    private final double[][] X;
    private final double[][] Y;
    private final int nClasses;

    // 3D view configuration
    private final int plotSize = 600;
    private final int padding = 80;
    
    // 3D rotation angles (in radians)
    private double rotationX = 0.5; // rotation around X axis
    private double rotationY = 0.3; // rotation around Y axis
    private double zoom = 1.2;
    
    // Mouse interaction
    private int lastMouseX, lastMouseY;
    private boolean isDragging = false;
    
    // Bounds in 3D space
    private double minX, maxX, minY, maxY, minZ, maxZ;
    private double centerX, centerY, centerZ;
    
    // Color palette
    private final Color[] regionColors;
    private final Color[] pointColors;
    
    // Cached 3D decision boundary visualization
    private BufferedImage boundaryImage;
    private static final int VOXEL_RESOLUTION = 30; // lower for performance, higher for quality
    private double lastRotationX = Double.NaN;
    private double lastRotationY = Double.NaN;
    private double lastZoom = Double.NaN;

    public BoundaryPanel3DTrue(ANN ann, double[][] X, double[][] Y, int nClasses) {
        this.ann = ann;
        this.X = X;
        this.Y = Y;
        this.nClasses = nClasses;

        this.regionColors = makeRegionPalette(nClasses);
        this.pointColors = makePointPalette(nClasses);

        computeBounds();
        setupMouseControls();
        
        setPreferredSize(new Dimension(plotSize + 2 * padding, plotSize + 2 * padding));
        setBackground(Color.WHITE);
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

        // Add margin
        double mx = (maxX - minX) * 0.30 + 1e-9;
        double my = (maxY - minY) * 0.30 + 1e-9;
        double mz = (maxZ - minZ) * 0.30 + 1e-9;
        minX -= mx; maxX += mx;
        minY -= my; maxY += my;
        minZ -= mz; maxZ += mz;

        centerX = (minX + maxX) / 2.0;
        centerY = (minY + maxY) / 2.0;
        centerZ = (minZ + maxZ) / 2.0;
    }

    private void setupMouseControls() {
        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                if (SwingUtilities.isLeftMouseButton(e)) {
                    isDragging = true;
                    lastMouseX = e.getX();
                    lastMouseY = e.getY();
                }
            }

            @Override
            public void mouseReleased(MouseEvent e) {
                if (SwingUtilities.isLeftMouseButton(e)) {
                    isDragging = false;
                }
            }
        });

        addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                if (isDragging && SwingUtilities.isLeftMouseButton(e)) {
                    int dx = e.getX() - lastMouseX;
                    int dy = e.getY() - lastMouseY;
                    
                    // Rotation sensitivity
                    rotationY += dx * 0.01;
                    rotationX += dy * 0.01;
                    
                    // Clamp X rotation to avoid gimbal lock
                    rotationX = Math.max(-Math.PI / 2 + 0.1, Math.min(Math.PI / 2 - 0.1, rotationX));
                    
                    lastMouseX = e.getX();
                    lastMouseY = e.getY();
                    
                    repaint();
                }
            }
        });
        
        // Mouse wheel for zoom
        addMouseWheelListener(e -> {
            double zoomFactor = 1.0 - e.getWheelRotation() * 0.1;
            zoom *= zoomFactor;
            zoom = Math.max(0.5, Math.min(3.0, zoom));
            repaint();
        });
    }

    /**
     * Project a 3D point to 2D screen coordinates using rotation and perspective projection.
     */
    private Point3D project3D(double x, double y, double z) {
        // Translate to center
        x -= centerX;
        y -= centerY;
        z -= centerZ;
        
        // Rotate around Y axis (horizontal rotation)
        double cosY = Math.cos(rotationY);
        double sinY = Math.sin(rotationY);
        double newX = x * cosY - z * sinY;
        double newZ = x * sinY + z * cosY;
        
        // Rotate around X axis (vertical rotation)
        double cosX = Math.cos(rotationX);
        double sinX = Math.sin(rotationX);
        double finalY = y * cosX - newZ * sinX;
        double finalZ = y * sinX + newZ * cosX;
        
        // Scale to fit viewport
        double range = Math.max(Math.max(maxX - minX, maxY - minY), maxZ - minZ);
        double scale = (plotSize * 0.8) / range * zoom;
        
        // Project to 2D (orthographic projection - simpler than perspective)
        double screenX = newX * scale;
        double screenY = finalY * scale;
        
        // Convert to pixel coordinates (origin at center of panel)
        int px = padding + plotSize / 2 + (int) screenX;
        int py = padding + plotSize / 2 - (int) screenY; // Y is flipped in screen coordinates
        
        // Store Z for depth sorting
        return new Point3D(px, py, finalZ * scale);
    }

    private static class Point3D {
        final int x, y;
        final double z; // depth for sorting
        
        Point3D(int x, int y, double z) {
            this.x = x;
            this.y = y;
            this.z = z;
        }
    }

    /**
     * Rebuild the 3D decision boundary visualization based on current rotation.
     * This samples the 3D space and creates a voxel-based visualization.
     */
    private void rebuildBoundary() {
        boundaryImage = new BufferedImage(plotSize, plotSize, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = boundaryImage.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setComposite(AlphaComposite.SrcOver);
        
        // Sample 3D space and project points
        double stepX = (maxX - minX) / VOXEL_RESOLUTION;
        double stepY = (maxY - minY) / VOXEL_RESOLUTION;
        double stepZ = (maxZ - minZ) / VOXEL_RESOLUTION;
        
        // Store projected points for depth sorting
        java.util.List<Voxel> voxels = new java.util.ArrayList<>();
        
        for (int iz = 0; iz <= VOXEL_RESOLUTION; iz++) {
            double z = minZ + iz * stepZ;
            for (int iy = 0; iy <= VOXEL_RESOLUTION; iy++) {
                double y = minY + iy * stepY;
                for (int ix = 0; ix <= VOXEL_RESOLUTION; ix++) {
                    double x = minX + ix * stepX;
                    
                    // Get prediction from ANN
                    double[] out = ann.apply(new double[]{x, y, z});
                    int cls = argMax(out);
                    double confidence = out[cls];
                    
                    // Project to 2D (adjusted for padding)
                    Point3D proj = project3D(x, y, z);
                    
                    int px = proj.x - padding;
                    int py = proj.y - padding;
                    
                    if (px >= 0 && px < plotSize && py >= 0 && py < plotSize) {
                        voxels.add(new Voxel(px, py, proj.z, cls, confidence));
                    }
                }
            }
        }
        
        // Sort by depth (back to front for proper rendering)
        voxels.sort((a, b) -> Double.compare(b.z, a.z));
        
        // Draw voxels
        for (Voxel v : voxels) {
            Color base = regionColors[v.cls % regionColors.length];
            int alpha = (int) clamp(50 + v.confidence * 100, 0, 150);
            Color c = new Color(base.getRed(), base.getGreen(), base.getBlue(), alpha);
            g.setColor(c);
            g.fillRect(v.x, v.y, 3, 3);
        }
        
        g.dispose();
        
        // Cache current rotation/zoom
        lastRotationX = rotationX;
        lastRotationY = rotationY;
        lastZoom = zoom;
    }
    
    private boolean needsRebuild() {
        // Rebuild if rotation or zoom changed significantly
        return Double.isNaN(lastRotationX) || 
               Double.isNaN(lastRotationY) ||
               Double.isNaN(lastZoom) ||
               Math.abs(rotationX - lastRotationX) > 0.01 ||
               Math.abs(rotationY - lastRotationY) > 0.01 ||
               Math.abs(zoom - lastZoom) > 0.05;
    }

    private static class Voxel {
        final int x, y;
        final double z;
        final int cls;
        final double confidence;
        
        Voxel(int x, int y, double z, int cls, double confidence) {
            this.x = x;
            this.y = y;
            this.z = z;
            this.cls = cls;
            this.confidence = confidence;
        }
    }

    @Override
    protected void paintComponent(Graphics g0) {
        super.paintComponent(g0);
        Graphics2D g = (Graphics2D) g0.create();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        // Rebuild boundary if rotation/zoom changed
        if (needsRebuild()) {
            rebuildBoundary();
        }

        // Draw decision boundary (voxel visualization)
        if (boundaryImage != null) {
            g.drawImage(boundaryImage, padding, padding, null);
        }

        // Draw axes
        drawAxes(g);

        // Draw training points (in front)
        drawTrainingPoints(g);

        // Draw legend and instructions
        drawInfo(g);

        g.dispose();
    }

    private void drawAxes(Graphics2D g) {
        g.setStroke(new BasicStroke(2));
        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 14));
        
        // Axis length
        double axisLen = Math.max(Math.max(maxX - minX, maxY - minY), maxZ - minZ) * 0.3;
        
        // Project axis endpoints
        Point3D origin = project3D(centerX, centerY, centerZ);
        Point3D xEnd = project3D(centerX + axisLen, centerY, centerZ);
        Point3D yEnd = project3D(centerX, centerY + axisLen, centerZ);
        Point3D zEnd = project3D(centerX, centerY, centerZ + axisLen);
        
        // Only draw if origin is visible
        if (origin.x < padding || origin.x > padding + plotSize || 
            origin.y < padding || origin.y > padding + plotSize) {
            return;
        }
        
        // Draw X axis (red)
        g.setColor(Color.RED);
        g.drawLine(origin.x, origin.y, xEnd.x, xEnd.y);
        g.fillOval(xEnd.x - 3, xEnd.y - 3, 6, 6);
        g.drawString("X", xEnd.x + 5, xEnd.y);
        
        // Draw Y axis (green)
        g.setColor(Color.GREEN);
        g.drawLine(origin.x, origin.y, yEnd.x, yEnd.y);
        g.fillOval(yEnd.x - 3, yEnd.y - 3, 6, 6);
        g.drawString("Y", yEnd.x + 5, yEnd.y);
        
        // Draw Z axis (blue)
        g.setColor(Color.BLUE);
        g.drawLine(origin.x, origin.y, zEnd.x, zEnd.y);
        g.fillOval(zEnd.x - 3, zEnd.y - 3, 6, 6);
        g.drawString("Z", zEnd.x + 5, zEnd.y);
    }

    private void drawTrainingPoints(Graphics2D g) {
        // Project all points
        java.util.List<PointWithClass> points = new java.util.ArrayList<>();
        for (int i = 0; i < X.length; i++) {
            Point3D proj = project3D(X[i][0], X[i][1], X[i][2]);
            int cls = argMax(Y[i]);
            points.add(new PointWithClass(proj, cls));
        }
        
        // Sort by depth
        points.sort((a, b) -> Double.compare(b.p.z, a.p.z));
        
        // Draw from back to front
        for (PointWithClass pc : points) {
            if (pc.p.x >= padding && pc.p.x < padding + plotSize &&
                pc.p.y >= padding && pc.p.y < padding + plotSize) {
                g.setColor(pointColors[pc.cls % pointColors.length]);
                g.fillOval(pc.p.x - 6, pc.p.y - 6, 12, 12);
                g.setColor(Color.BLACK);
                g.setStroke(new BasicStroke(1.5f));
                g.drawOval(pc.p.x - 6, pc.p.y - 6, 12, 12);
            }
        }
    }

    private static class PointWithClass {
        final Point3D p;
        final int cls;
        
        PointWithClass(Point3D p, int cls) {
            this.p = p;
            this.cls = cls;
        }
    }

    private void drawInfo(Graphics2D g) {
        // Legend
        int x = padding + 10;
        int y = padding + 10;
        int boxW = 180;
        int boxH = 18 * (nClasses + 1) + 10;

        g.setColor(new Color(255, 255, 255, 220));
        g.fillRoundRect(x, y, boxW, boxH, 10, 10);
        g.setColor(Color.DARK_GRAY);
        g.setStroke(new BasicStroke(1));
        g.drawRoundRect(x, y, boxW, boxH, 10, 10);

        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 12));
        g.setColor(Color.BLACK);
        g.drawString("Classes", x + 10, y + 18);

        g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 11));
        for (int c = 0; c < nClasses; c++) {
            int yy = y + 18 + 18 * (c + 1);
            g.setColor(pointColors[c % pointColors.length]);
            g.fillOval(x + 10, yy + 2, 10, 10);
            g.setColor(Color.BLACK);
            g.drawOval(x + 10, yy + 2, 10, 10);
            g.drawString("Class " + c, x + 30, yy + 12);
        }
        
        // Instructions
        x = padding + 10;
        y = padding + plotSize - 50;
        g.setColor(new Color(255, 255, 255, 220));
        g.fillRoundRect(x, y, 250, 40, 10, 10);
        g.setColor(Color.DARK_GRAY);
        g.drawRoundRect(x, y, 250, 40, 10, 10);
        g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 10));
        g.setColor(Color.BLACK);
        g.drawString("Left drag: Rotate | Wheel: Zoom", x + 10, y + 18);
        g.drawString("Axes: Red=X, Green=Y, Blue=Z", x + 10, y + 32);
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

    