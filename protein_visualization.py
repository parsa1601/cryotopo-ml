"""
Protein Visualization Module

This module contains all plotting and visualization functionality for protein structures,
including SVM models and 3D cylindrical structures (alpha-helices and beta-strands).
"""

import numpy as np
import matplotlib

try:
    matplotlib.use("Qt5Agg")
    print("Using Qt5Agg backend for interactive 3D viewing")
except ImportError:
    try:
        matplotlib.use("TkAgg")
        print("Using TkAgg backend for interactive 3D viewing")
    except ImportError:
        try:
            matplotlib.use("GTK3Agg")
            print("Using GTK3Agg backend for interactive 3D viewing")
        except ImportError:
            matplotlib.use("Agg")
            print(
                "Warning: No interactive backend available, using Agg (non-interactive)"
            )

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class ProteinVisualizer:
    """
    A class for visualizing protein structures and machine learning models in 3D space.
    """

    def __init__(self):
        pass

    def plot_3d_cylindrical_structures_with_svm(
        self,
        X_train,
        y_train,
        protein_name,
        structure_type="Helix",
        svm_model=None,
        title_suffix="",
    ):
        """
        Plot 3D cylindrical structures (alpha-helices or beta-strands) with direction vectors
        and optional SVM decision boundaries in the same visualization.

        Parameters:
        X_train: 3D training data points (n_samples, 3)
        y_train: Training labels
        protein_name: Name of the protein for the plot title
        structure_type: "Helix" or "Strand"
        svm_model: Optional trained SVM model to show decision boundaries
        title_suffix: Additional text for the plot title
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

        unique_classes = np.unique(y_train)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))

        model_info = f" with SVM {title_suffix}" if svm_model else ""
        print(
            f"Creating 3D cylindrical visualization for {protein_name} with {len(unique_classes)} {structure_type.lower()}s{model_info}"
        )

        for i, class_label in enumerate(unique_classes):
            mask = y_train == class_label
            class_points = X_train[mask]

            if len(class_points) < 2:
                continue

            ax.scatter(
                class_points[:, 0],
                class_points[:, 1],
                class_points[:, 2],
                c=[colors[i]],
                label=f"{structure_type} {class_label}",
                s=50,
                alpha=0.8,
            )

            cylinder_info = self._calculate_cylinder_properties(class_points)

            self._draw_cylinder(ax, cylinder_info, colors[i], alpha=0.2)

            self._draw_direction_vector(ax, cylinder_info, colors[i])

            ax.text(
                class_points[0, 0],
                class_points[0, 1],
                class_points[0, 2],
                f"{class_label}-Start",
                fontsize=8,
                color=colors[i],
            )
            ax.text(
                class_points[-1, 0],
                class_points[-1, 1],
                class_points[-1, 2],
                f"{class_label}-End",
                fontsize=8,
                color=colors[i],
            )

        if svm_model is not None:
            try:
                self._plot_hyperplane(ax, X_train, svm_model)

                if hasattr(svm_model, "support_vectors_"):
                    support_vectors = svm_model.support_vectors_
                    ax.scatter(
                        support_vectors[:, 0],
                        support_vectors[:, 1],
                        support_vectors[:, 2],
                        s=100,
                        facecolors="none",
                        edgecolors="black",
                        linewidth=2,
                        label="Support Vectors",
                    )

                for i in range(0, len(X_train), max(1, len(X_train) // 30)):
                    ax.text(
                        X_train[i, 0],
                        X_train[i, 1],
                        X_train[i, 2],
                        str(y_train[i]),
                        fontsize=6,
                        alpha=0.4,
                    )

            except Exception as e:
                print(f"Could not plot SVM decision boundaries: {e}")

        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_zlabel("Z Coordinate")

        title = f"3D {structure_type} Structures - {protein_name}"
        if svm_model:
            title += f" {title_suffix}"
        ax.set_title(title)
        ax.legend()

        ax.view_init(elev=20, azim=45)

        plt.tight_layout()

        backend = matplotlib.get_backend()
        print(f"Displaying interactive 3D plot (backend: {backend})")
        print("🎮 You can rotate, zoom, and pan the 3D plot!")
        print("   Close the plot window to continue...")
        plt.show()

    def _plot_hyperplane(self, ax, X_train, svm_model):
        """
        Helper method to plot decision boundaries for SVM in 3D space as hyperplanes.
        """
        try:
            x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
            y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
            z_min, z_max = X_train[:, 2].min() - 1, X_train[:, 2].max() + 1

            if hasattr(svm_model, "coef_") and hasattr(svm_model, "intercept_"):
                coef = svm_model.coef_
                intercept = svm_model.intercept_

                self._plot_multiclass_hyperplanes(
                    ax, coef, intercept, x_min, x_max, y_min, y_max, z_min, z_max
                )

        except Exception as e:
            print(f"Could not plot decision boundaries: {e}")

    def _plot_multiclass_hyperplanes(
        self,
        ax,
        coef_matrix,
        intercept_vector,
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
    ):
        """
        Plot up to 5 hyperplanes for classification.
        """
        colors = ["red", "blue", "green", "orange", "purple"]

        max_hyperplanes = min(5, len(coef_matrix))

        for i in range(max_hyperplanes):
            coef = coef_matrix[i]
            intercept = intercept_vector[i]
            color = colors[i]

            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 15), np.linspace(y_min, y_max, 15)
            )

            if abs(coef[2]) > 1e-6:
                zz = -(coef[0] * xx + coef[1] * yy + intercept) / coef[2]
                zz = np.clip(zz, z_min, z_max)

                ax.plot_surface(
                    xx, yy, zz, alpha=0.2, color=color, label=f"Hyperplane {i + 1}"
                )

        print(f"Plotted {max_hyperplanes} hyperplanes for classification")

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
        axis_unit = (
            axis_vector / axis_length if axis_length > 0 else np.array([0, 0, 1])
        )

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
            "start": center_start,
            "end": center_end,
            "axis_vector": axis_vector,
            "axis_unit": axis_unit,
            "radius": radius,
            "length": axis_length,
        }

    def _draw_cylinder(self, ax, cylinder_info, color, alpha=0.3):
        """
        Draw a 3D cylinder using the calculated properties.
        """
        start = cylinder_info["start"]
        end = cylinder_info["end"]
        radius = cylinder_info["radius"]
        axis_unit = cylinder_info["axis_unit"]

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
        theta = np.linspace(0, 2 * np.pi, 20)

        # Bottom circle
        bottom_circle = np.array(
            [
                start
                + radius * (np.cos(t) * perpendicular1 + np.sin(t) * perpendicular2)
                for t in theta
            ]
        )

        # Top circle
        top_circle = np.array(
            [
                end + radius * (np.cos(t) * perpendicular1 + np.sin(t) * perpendicular2)
                for t in theta
            ]
        )

        # Draw the cylinder surface
        for j in range(len(theta) - 1):
            # Create quadrilateral for cylinder surface
            quad = [
                bottom_circle[j],
                bottom_circle[j + 1],
                top_circle[j + 1],
                top_circle[j],
            ]
            poly = [quad]
            collection = Poly3DCollection(
                poly, alpha=alpha, facecolor=color, edgecolor="none"
            )
            ax.add_collection3d(collection)

        # Draw top and bottom circles
        bottom_poly = [bottom_circle.tolist()]
        top_poly = [top_circle.tolist()]
        ax.add_collection3d(
            Poly3DCollection(
                bottom_poly, alpha=alpha, facecolor=color, edgecolor="none"
            )
        )
        ax.add_collection3d(
            Poly3DCollection(top_poly, alpha=alpha, facecolor=color, edgecolor="none")
        )

    def _draw_direction_vector(self, ax, cylinder_info, color):
        """
        Draw an arrow showing the direction vector through the center of the cylinder.
        """
        start = cylinder_info["start"]
        end = cylinder_info["end"]

        ax.quiver(
            start[0],
            start[1],
            start[2],
            end[0] - start[0],
            end[1] - start[1],
            end[2] - start[2],
            color=color,
            arrow_length_ratio=0.1,
            linewidth=3,
            alpha=0.8,
        )

        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=color,
            linewidth=2,
            alpha=0.9,
            linestyle="--",
        )
