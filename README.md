# ML-based Secondary Structure Element (SSE) Matching and Direction Detection

## Overview

This system implements a machine learning approach for detecting structural direction of protein secondary structure elements (SSEs) using alpha carbon coordinates and Dynamic Time Warping (DTW). The methodology demonstrates direct coordinate-based geometric learning for SSE matching in cryo-EM density maps.

## Core Methodology

- **Input**: Alpha carbon coordinates (3D Cα sequences) from helices/strands and SSETracer-detected segments
- **Features**: Flattened 3N-dimensional coordinate vectors with normalization and fixed-length resampling (20-30 points)
- **Learning**: Direct coordinate-based geometric learning without hand-crafted features
- **Validation**: Two-stage filtering through ML accuracy and ground truth topology availability

## System Architecture

### Key Components
- **`protein_trainer.py`**: Main orchestrator coordinating the entire workflow
- **`ml_classifiers.py`**: ML algorithms (SVM Linear/RBF, Random Forest, 1-NN KNN)
- **`direction_analyzer.py`**: DTW-based direction detection
- **`hyperparameter_optimizer.py`**: Grid search and parameter optimization
- **`data_loader.py`**: Data loading and preprocessing
- **`main.py`**: Entry point

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

The system uses Dynamic Time Warping (DTW) for direction detection:

1. **Forward Test**: DTW distance between stick and model coordinates
2. **Backward Test**: DTW distance between reversed stick and model coordinates
3. **Direction Decision**: Lower DTW distance determines direction (1 = same, -1 = opposite)

**Workflow**: Data loading → ML training → mapping prediction → DTW direction detection → validation against ground truth

## Machine Learning Pipeline

### Process
1. Extract normalized Cα coordinates → flatten to 3N-dimensional vectors
2. Train classifiers to map coordinate patterns to helix/strand IDs
3. Classify stick coordinates and apply DTW direction detection
4. Evaluate using classification accuracy and direction detection accuracy

### Hyperparameter Optimization
- **Grid Search Mode**: Exhaustive optimization across all algorithms and parameters
```
For each protein:
  For each algorithm:
    For each parameter combination:
      - Train model with parameters
      - Evaluate on test data using mapped accuracy
      - Track performance across proteins
  
After all proteins processed:
  - Calculate average accuracy for each parameter set
  - Select globally best parameters
  - Save to JSON file for future use
```
- **Pre-optimized Mode**: Uses saved best parameters from `best_hyperparameters.json`

## Key Innovations
- **Direct Coordinate Learning**: Achieves high SSE matching accuracy using only normalized alpha carbon coordinates without hand-crafted geometric features
- **DTW-Based Direction Detection**: Combines ML classification with Dynamic Time Warping for robust, data-driven direction detection\
  