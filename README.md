[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18911570.svg)](https://doi.org/10.5281/zenodo.18911570)

# ML-based Secondary Structure Element (SSE) Matching and Direction Detection

## Overview

This system implements a machine learning approach for detecting structural direction of protein secondary structure elements (SSEs) using alpha carbon coordinates and Dynamic Time Warping (DTW). The methodology demonstrates direct coordinate-based geometric learning for SSE matching in cryo-EM density maps.

## Quick Start

### Prerequisites
- Python 3.8 or higher

### Installation Steps

1. **Download the Repository**
  Extract the source code ZIP file and cd to the extracted directory.

2. **Create a Virtual Environment**
   ```cmd
   python -m venv venv
   ```

3. **Activate the Virtual Environment**
   ```cmd
   venv\Scripts\activate
   ```
   You should see `(venv)` prefix in your command prompt.

4. **Install Required Dependencies**
   ```cmd
   pip install -r requirements.txt
   ```

### Configuration and Data Preparation

Before running the analysis, ensure the following:

1. **Specify Protein Lists**
  - Edit `config.py` and adjust the `HELIX_PROTEIN_LIST` and `STRAND_PROTEIN_LIST` variables to include the protein IDs you wish to train and analyze.

2. **Prepare Data Folder**
  - Copy the `Archive` folder (containing your protein data) into the current working directory.
  - The folder structure should match the format described in the [Data Structure and File Organization](#data-structure-and-file-organization) section above.
  

### Running the Analysis

### Performance Analysis (Helix + Strand)
```cmd
python sse_matching/main.py
```
This will:
- Train ML models on helix and strand data
- Perform direction detection using DTW
- Generate accuracy reports and visualizations

#### Output Files
After running the analysis, you'll find:
- `Final_Results.json` - Complete results in JSON format
- `analytical_report.txt` - Detailed performance metrics per algorithm
- Various accuracy and performance charts:
    - `f1_measure_chart.png`
    - `protein_error_rate_line_chart.png`
    - `protein_metrics_bar_chart.png`

### Direction analysis with the best algorithm (Helix + Strand)
If you have the `Final_Results.json` file from a previous run, you can perform direction analysis using the best algorithm identified in the results:

```cmd
python sse_matching/main.py --direction-analysis
```
This will generate the `direction_analysis_report.txt` file. 

### LPTD Comparison (Helix + Strand)
To compare the performance and runtime of the LPTD method against the best ML algorithm (SVM RBF):

```cmd
python sse_matching/main.py --lptd-comparison
```
This will:
- Run the ML training pipeline to get baseline metrics
- Run the LPTD method on the same dataset
- Generate a runtime comparison chart (`runtime_comparison_chart.png`)
- Generate accuracy charts including LPTD results


## Visualizations & Plot Generation

This repository includes a dedicated Jupyter Notebook, **all_figures.ipynb**, which serves as a centralized hub for generating all project figures and plots.

### Key Features of the Notebook

#### Automated Charting

The notebook cells read directly from the generated output files (**Final_Results.json** and **Protein_List.xlsx**) to automatically create high-quality, formatted charts. These include:

* Performance comparison charts
* Runtime charts
* Grouped bar charts for **F1-measures**

#### 3D Protein Visualization (5I1M)

The notebook includes dedicated cells for rendering **3D scatter and line plots** of the protein **5I1M**. This allows you to visually map and inspect both its extracted **helices** and **strands (sticks)** directly in your browser.

### How to Use the Notebook

1. Ensure your **virtual environment** is active and the required libraries are installed:

```
pip install pandas matplotlib jupyter
```

2. Launch the notebook environment:

```
jupyter notebook
```

or

```
jupyter lab
```

3. Open **all_figures.ipynb**.

4. Run the notebook cells **sequentially** after your main analysis has finished exporting the **JSON** and **Excel** files.


## Key Components
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


## Hyperparameter Optimization
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
