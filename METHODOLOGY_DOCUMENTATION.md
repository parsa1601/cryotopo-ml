# Protein Structure Direction Detection Methodology

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Data Structure and File Organization](#data-structure-and-file-organization)
3. [Direction Detection Methodology](#direction-detection-methodology)
4. [Machine Learning Pipeline](#machine-learning-pipeline)

## System Architecture

The system consists of several key components:

### Core Classes
- **ProteinAssignmentUsingMultipleML**: Main class handling ML algorithms and direction analysis
- **ProteinVisualizer**: 3D visualization of protein structures

### Machine Learning Algorithms
1. **SVM Linear**: Linear Support Vector Machine
2. **SVM RBF**: Radial Basis Function SVM
3. **Random Forest**: Ensemble method with 100 estimators

## Data Structure and File Organization

### Input Files Required for Each Protein
For each protein (e.g., `1A7D`), the system expects:

#### Helix Mode
```
Archive/
├── {PROTEIN}/
│   ├── {PROTEIN}_Helices.csv    # Training data (helix coordinates + labels)
│   ├── {PROTEIN}_Sticks.csv     # Test data (stick coordinates + labels)
│   └── {PROTEIN}_Topology.csv   # Mapping and direction information
```

#### Strand Mode
```
Archive/
├── STRANDS/
│   └── {PROTEIN}/
│       └── Sheet/
│           ├── {PROTEIN}_Strands.csv         # Training data
│           ├── {PROTEIN}_Sticks_Strands.csv  # Test data
│           └── {PROTEIN}_Topology.csv        # Mapping and direction information
```

### Topology File Format

The topology file is crucial for direction detection and contains three columns:

```csv
train_label,test_label,direction
1,0,           # No direction info (NaN)
2,1,-1         # Stick 1 maps to helix 2, opposite direction
3,2,-1         # Stick 2 maps to helix 3, opposite direction
4,3,1          # Stick 3 maps to helix 4, same direction
5,4,1          # Stick 4 maps to helix 5, same direction
6,0,           # No direction info (NaN)
```

**Column Meanings:**
- **Column 1**: Train label (helix/strand ID)
- **Column 2**: Test label (stick ID)
- **Column 3**: Direction indicator
  - `1`: Same direction as the model structure
  - `-1`: Opposite direction to the model structure
  - `NaN` or empty: No direction information available

## Direction Detection Methodology

### Dynamic Time Warping (DTW) Algorithm

The system uses DTW to determine the optimal direction of sticks by comparing them with model coordinates.

### Direction Determination Process

1. **Forward Direction Test**: Compare stick coordinates directly with model coordinates
2. **Backward Direction Test**: Compare reversed stick coordinates with model coordinates
3. **Decision**: Choose direction with lower DTW distance
   - Returns `1` if forward distance ≤ backward distance
   - Returns `-1` if backward distance < forward distance

### Direction Analysis Workflow

For each protein, the system:

1. **Loads Data**: Reads helix/strand coordinates and stick coordinates
2. **Trains ML Models**: Uses helix/strand data to train classifiers
3. **Predicts Mappings**: Predicts which helix/strand each stick corresponds to
4. **Validates Predictions**: Only analyzes directions for correctly predicted mappings
5. **Applies DTW**: Uses DTW to detect actual direction
6. **Compares with Ground Truth**: Compares detected direction with topology file direction

**Key Bottlenecks:**
1. **ML Accuracy Gate**: Only correctly classified sticks proceed to direction analysis
2. **Direction Data Gate**: Only sticks with direction information in topology are analyzed

## Machine Learning Pipeline

### Training Process

1. **Data Loading**: Load helix/strand coordinates (training data)
2. **Label Processing**: Extract class labels from 4th column
3. **Model Training**: Train candidate ML models on helix/strand coordinates
4. **Validation**: Test on stick coordinates (test data)

### Evaluation Metrics

- **Classification Accuracy**: Percentage of correctly mapped sticks to helices/strands
- **Direction Detection Accuracy**: Percentage of correctly detected directions (subset of correctly classified sticks)

### Best Algorithm Selection

The system automatically selects the best-performing algorithm based on classification accuracy, then uses it for direction analysis.
