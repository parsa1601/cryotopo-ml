"""
Protein Visualization Module

This module contains all plotting and visualization functionality for protein structures,
including SVM models and 3D cylindrical structures (alpha-helices and beta-strands).
"""

import numpy as np
import matplotlib
# Try different interactive backends in order of preference
try:
    matplotlib.use('Qt5Agg')  # Try Qt5 first
    print("Using Qt5Agg backend for interactive 3D viewing")
except ImportError:
    try:
        matplotlib.use('TkAgg')  # Try Tk
        print("Using TkAgg backend for interactive 3D viewing")
    except ImportError:
        try:
            matplotlib.use('GTK3Agg')  # Try GTK3
            print("Using GTK3Agg backend for interactive 3D viewing")
        except ImportError:
            matplotlib.use('Agg')  # Fallback to non-interactive
            print("Warning: No interactive backend available, using Agg (non-interactive)")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class ProteinVisualizer:
    """
    A class for visualizing protein structures and machine learning models in 3D space.
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        pass
    
    def plot_3d_svm_model(self, X_train, y_train, svm_model, protein_name, title_suffix=""):
        """
        Plot the trained SVM model in 3D space with training data points, labels, and decision boundaries.
        
        Parameters:
        X_train: 3D training data points (n_samples, 3)
        y_train: Training labels
        svm_model: Trained SVM model
        protein_name: Name of the protein for the plot title
        title_suffix: Additional text for the plot title
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get unique classes and create color map
        unique_classes = np.unique(y_train)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))
        
        print(f"Plotting 3D model for {protein_name} {title_suffix} with {len(unique_classes)} classes")
        
        # Plot training data points with different colors for each class
        for i, class_label in enumerate(unique_classes):
            mask = y_train == class_label
            ax.scatter(X_train[mask, 0], X_train[mask, 1], X_train[mask, 2], 
                      c=[colors[i]], label=f'Class {class_label}', s=50, alpha=0.7)
        
        # Add text annotations for some points (to avoid clutter, show only every 10th point)
        for i in range(0, len(X_train), max(1, len(X_train)//20)):  # Show max 20 labels
            ax.text(X_train[i, 0], X_train[i, 1], X_train[i, 2], 
                   str(y_train[i]), fontsize=8, alpha=0.6)
        
        # For linear SVM, try to plot decision boundaries
        if hasattr(svm_model, 'coef_') and svm_model.coef_ is not None:
            try:
                self._plot_decision_boundaries(ax, X_train, svm_model)
            except Exception as e:
                print(f"Could not plot decision boundaries: {e}")
        
        # Highlight support vectors if available
        if hasattr(svm_model, 'support_vectors_'):
            support_vectors = svm_model.support_vectors_
            ax.scatter(support_vectors[:, 0], support_vectors[:, 1], support_vectors[:, 2], 
                      s=100, facecolors='none', edgecolors='black', linewidth=2, 
                      label='Support Vectors')
        
        # Set labels and title
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        ax.set_title(f'3D SVM Model - {protein_name} {title_suffix}')
        ax.legend()
        
        # Improve the view
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        
        # Force interactive display
        backend = matplotlib.get_backend()
        print(f"Displaying interactive SVM plot (backend: {backend})")
        print("🎮 You can rotate, zoom, and pan the 3D plot!")
        print("   Close the plot window to continue...")
        plt.show()
    
    def _plot_decision_boundaries(self, ax, X_train, svm_model):
        """
        Helper method to plot decision boundaries for SVM in 3D space.
        """
        # Create a 3D grid for visualization
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        z_min, z_max = X_train[:, 2].min() - 1, X_train[:, 2].max() + 1
        
        # Create a coarser grid to avoid memory issues
        resolution = 10
        xx, yy, zz = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution),
            np.linspace(z_min, z_max, resolution)
        )
        
        # Flatten the grid for prediction
        grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
        
        # Get decision function values
        if hasattr(svm_model, 'decision_function'):
            Z = svm_model.decision_function(grid_points)
            if Z.ndim > 1:  # Multi-class case
                # For multi-class, we can show the decision regions by coloring points
                predictions = svm_model.predict(grid_points)
                unique_preds = np.unique(predictions)
                colors_decision = plt.cm.viridis(np.linspace(0, 1, len(unique_preds)))
                
                for i, pred_class in enumerate(unique_preds):
                    mask = predictions == pred_class
                    if np.any(mask):
                        ax.scatter(grid_points[mask, 0], grid_points[mask, 1], grid_points[mask, 2],
                                 c=[colors_decision[i]], alpha=0.1, s=1)
                print(f"Plotted decision regions for {len(unique_preds)} classes")
            else:
                # Binary case - plot the decision boundary (Z ≈ 0)
                Z = Z.reshape(xx.shape)
                # Find points close to decision boundary
                boundary_mask = np.abs(Z) < 0.1
                if np.any(boundary_mask):
                    ax.scatter(xx[boundary_mask], yy[boundary_mask], zz[boundary_mask],
                             c='red', alpha=0.3, s=5, label='Decision Boundary')
                print("Plotted decision boundary for binary classification")
    
    def _plot_hyperplane(self, ax, X_train, svm_model):
        """
        Helper method to plot the hyperplane for linear SVM in 3D space.
        This method is kept for backward compatibility but _plot_decision_boundaries is more comprehensive.
        """
        self._plot_decision_boundaries(ax, X_train, svm_model)

    def plot_3d_cylindrical_structures(self, X_train, y_train, protein_name, structure_type="Helix"):
        """
        Plot 3D cylindrical structures (alpha-helices or beta-strands) with direction vectors.
        Each class of data points is wrapped in a cylinder with a vector showing the direction
        from first to last data point.
        
        Parameters:
        X_train: 3D training data points (n_samples, 3)
        y_train: Training labels
        protein_name: Name of the protein for the plot title
        structure_type: "Helix" or "Strand"
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get unique classes and create color map
        unique_classes = np.unique(y_train)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))
        
        print(f"Creating 3D cylindrical visualization for {protein_name} with {len(unique_classes)} {structure_type.lower()}s")
        
        for i, class_label in enumerate(unique_classes):
            mask = y_train == class_label
            class_points = X_train[mask]
            
            if len(class_points) < 2:
                continue
                
            # Plot the data points
            ax.scatter(class_points[:, 0], class_points[:, 1], class_points[:, 2], 
                      c=[colors[i]], label=f'{structure_type} {class_label}', s=50, alpha=0.8)
            
            # Calculate cylinder properties
            cylinder_info = self._calculate_cylinder_properties(class_points)
            
            # Draw the cylinder
            self._draw_cylinder(ax, cylinder_info, colors[i], alpha=0.2)
            
            # Draw the direction vector (axis of the cylinder)
            self._draw_direction_vector(ax, cylinder_info, colors[i])
            
            # Add labels for first and last points
            ax.text(class_points[0, 0], class_points[0, 1], class_points[0, 2], 
                   f'{class_label}-Start', fontsize=8, color=colors[i])
            ax.text(class_points[-1, 0], class_points[-1, 1], class_points[-1, 2], 
                   f'{class_label}-End', fontsize=8, color=colors[i])
        
        # Set labels and title
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        ax.set_title(f'3D {structure_type} Structures - {protein_name}')
        ax.legend()
        
        # Improve the view
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        
        # Force interactive display
        backend = matplotlib.get_backend()
        print(f"Displaying interactive 3D plot (backend: {backend})")
        print("🎮 You can rotate, zoom, and pan the 3D plot!")
        print("   Close the plot window to continue...")
        plt.show()
    
    def _calculate_cylinder_properties(self, points):
        """
        Calculate cylinder properties to wrap around a set of 3D points.
        Returns the axis direction, center line, and radius.
        """
        # Sort points to get proper sequence (first to last)
        # For now, we'll use the order they appear in the data
        start_point = points[0]
        end_point = points[-1]
        
        # Calculate the axis vector (direction from start to end)
        axis_vector = end_point - start_point
        axis_length = np.linalg.norm(axis_vector)
        axis_unit = axis_vector / axis_length if axis_length > 0 else np.array([0, 0, 1])
        
        # Calculate the center line of the cylinder
        center_start = start_point
        center_end = end_point
        
        # Calculate radius as the maximum distance from points to the axis line
        radius = 0
        for point in points:
            # Distance from point to the line (start_point to end_point)
            point_to_start = point - start_point
            projection_length = np.dot(point_to_start, axis_unit)
            projection_point = start_point + projection_length * axis_unit
            distance_to_axis = np.linalg.norm(point - projection_point)
            radius = max(radius, distance_to_axis)
        
        # Add some padding to the radius
        radius = max(radius * 1.2, 0.5)  # Minimum radius of 0.5
        
        return {
            'start': center_start,
            'end': center_end,
            'axis_vector': axis_vector,
            'axis_unit': axis_unit,
            'radius': radius,
            'length': axis_length
        }
    
    def _draw_cylinder(self, ax, cylinder_info, color, alpha=0.3):
        """
        Draw a 3D cylinder using the calculated properties.
        """
        start = cylinder_info['start']
        end = cylinder_info['end']
        radius = cylinder_info['radius']
        axis_unit = cylinder_info['axis_unit']
        
        # Create two perpendicular vectors to the axis
        # Find a vector not parallel to axis_unit
        if abs(axis_unit[2]) < 0.9:
            perpendicular1 = np.cross(axis_unit, [0, 0, 1])
        else:
            perpendicular1 = np.cross(axis_unit, [1, 0, 0])
        
        perpendicular1 = perpendicular1 / np.linalg.norm(perpendicular1)
        perpendicular2 = np.cross(axis_unit, perpendicular1)
        perpendicular2 = perpendicular2 / np.linalg.norm(perpendicular2)
        
        # Create circular cross-sections
        theta = np.linspace(0, 2*np.pi, 20)
        
        # Bottom circle
        bottom_circle = np.array([start + radius * (np.cos(t) * perpendicular1 + np.sin(t) * perpendicular2) 
                                 for t in theta])
        
        # Top circle
        top_circle = np.array([end + radius * (np.cos(t) * perpendicular1 + np.sin(t) * perpendicular2) 
                              for t in theta])
        
        # Draw the cylinder surface
        for j in range(len(theta)-1):
            # Create quadrilateral for cylinder surface
            quad = [bottom_circle[j], bottom_circle[j+1], top_circle[j+1], top_circle[j]]
            poly = [quad]
            collection = Poly3DCollection(poly, alpha=alpha, facecolor=color, edgecolor='none')
            ax.add_collection3d(collection)
        
        # Draw top and bottom circles
        bottom_poly = [bottom_circle.tolist()]
        top_poly = [top_circle.tolist()]
        ax.add_collection3d(Poly3DCollection(bottom_poly, alpha=alpha, facecolor=color, edgecolor='none'))
        ax.add_collection3d(Poly3DCollection(top_poly, alpha=alpha, facecolor=color, edgecolor='none'))
    
    def _draw_direction_vector(self, ax, cylinder_info, color):
        """
        Draw an arrow showing the direction vector through the center of the cylinder.
        """
        start = cylinder_info['start']
        end = cylinder_info['end']
        
        # Draw the main axis vector as an arrow
        ax.quiver(start[0], start[1], start[2], 
                 end[0] - start[0], end[1] - start[1], end[2] - start[2],
                 color=color, arrow_length_ratio=0.1, linewidth=3, alpha=0.8)
        
        # Add a thicker line for better visibility
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
               color=color, linewidth=2, alpha=0.9, linestyle='--')
