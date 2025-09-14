#!/usr/bin/env python3
"""
Interactive 3D Viewer for Protein Structure Analysis

This script provides interactive 3D visualization of protein structures
with cylindrical representations and SVM decision boundaries.
"""

import os
import sys
import numpy as np
import pandas as pd

# Force Qt platform to prevent threading issues
os.environ['QT_API'] = 'pyqt5'
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# Import Qt first to initialize properly
try:
    from PyQt5.QtWidgets import QApplication
    import matplotlib
    matplotlib.use('Qt5Agg', force=True)
    import matplotlib.pyplot as plt
    
    # Create QApplication instance if it doesn't exist
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    print("✓ Qt5 successfully initialized for interactive viewing")
    
except ImportError:
    print("❌ PyQt5 not available. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyQt5"])
    
    # Try again after installation
    from PyQt5.QtWidgets import QApplication
    import matplotlib
    matplotlib.use('Qt5Agg', force=True)
    import matplotlib.pyplot as plt
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

# Import after Qt setup
from protein_visualization import ProteinVisualizer


def create_simple_interactive_plot():
    """Create a simple interactive 3D plot to test the setup"""
    
    print("🧪 Testing interactive 3D plotting...")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create sample data
    n = 100
    x = np.random.randn(n)
    y = np.random.randn(n)
    z = np.random.randn(n)
    colors = np.random.rand(n)
    
    # Create 3D scatter plot
    scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', s=50, alpha=0.7)
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Interactive 3D Test Plot\n(Use mouse to rotate, zoom, pan)')
    
    # Add colorbar
    plt.colorbar(scatter)
    
    print("✓ Showing interactive plot - you can rotate, zoom, and pan!")
    print("🎮 Close the plot window to continue...")
    plt.show()


def interactive_protein_viewer():
    """Launch interactive 3D viewer for protein structures"""
    
    print("="*70)
    print("🧬 INTERACTIVE 3D PROTEIN STRUCTURE VIEWER")
    print("="*70)
    
    backend = matplotlib.get_backend()
    print(f"🎯 Backend: {backend}")
    print("✓ Interactive mode ready!")
    print("🎮 Controls: Left mouse=rotate, Right mouse=zoom, Middle=pan")
    print("="*70)
    
    try:
        # Initialize visualizer
        visualizer = ProteinVisualizer()
        
        # Load sample protein data
        protein_name = "1A7D"
        csv_path = "Archive/"
        
        print(f"📊 Loading protein data for {protein_name}...")
        
        helix_records = f"{csv_path}/{protein_name}/{protein_name}_Helices.csv"
        if not os.path.exists(helix_records):
            print(f"❌ Data file not found: {helix_records}")
            print("   Make sure you're in the correct directory.")
            return
            
        helix_df = pd.read_csv(helix_records, header=None)
        X_train = helix_df.iloc[:, :3].to_numpy()
        y_train = helix_df.iloc[:, 3].to_numpy().astype(int)
        
        print(f"✓ Loaded {len(X_train)} data points")
        print(f"✓ Found {len(np.unique(y_train))} helices: {sorted(np.unique(y_train))}")
        
        # Create 3D cylindrical visualization
        print("\n🔬 Creating interactive 3D cylindrical visualization...")
        visualizer.plot_3d_cylindrical_structures(X_train, y_train, protein_name, "Helix")
        
        # Interactive menu
        while True:
            print("\n🎯 Options:")
            print("1. Create another protein visualization")
            print("2. Test simple 3D plot")
            print("3. Exit")
            
            try:
                choice = input("Enter choice (1-3): ").strip()
                
                if choice == '1':
                    proteins = ["1A7D", "6EM3", "1BZ4", "1HG5"]
                    print(f"Available proteins: {', '.join(proteins)}")
                    new_protein = input("Enter protein name: ").strip().upper()
                    
                    if new_protein in proteins:
                        try:
                            new_file = f"{csv_path}/{new_protein}/{new_protein}_Helices.csv"
                            new_df = pd.read_csv(new_file, header=None)
                            new_X = new_df.iloc[:, :3].to_numpy()
                            new_y = new_df.iloc[:, 3].to_numpy().astype(int)
                            visualizer.plot_3d_cylindrical_structures(new_X, new_y, new_protein, "Helix")
                        except FileNotFoundError:
                            print(f"❌ Data not found for {new_protein}")
                    else:
                        print("❌ Invalid protein name")
                
                elif choice == '2':
                    create_simple_interactive_plot()
                
                elif choice == '3':
                    print("👋 Goodbye!")
                    break
                
                else:
                    print("❌ Invalid choice")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
            
    except FileNotFoundError as e:
        print(f"❌ File error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        create_simple_interactive_plot()
    else:
        interactive_protein_viewer()
