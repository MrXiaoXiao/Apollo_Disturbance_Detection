# Apollo_Disturbance_Detection

Apollo Annotator is a user-friendly application for annotating and analyzing recently re-archived Apollo lunar seismic data (Nunn et al., 2020) in MiniSEED format. This tool offers both manual and automated workflows to help with efficiently process and analyze.

## Table of Contents

- [Installation](#installation)
- [Manual Annotation](#manual-annotation)
- [Automated Workflow](#automated-workflow)

## Installation

Set up the required environment using conda and pip:

```bash
conda create -n apollo_annotator
conda activate apollo_annotator
conda install python=3.9
pip install tensorflow==2.15 keras==2.15 obspy==1.4 astropy==5.3 jupyter notebook pyyaml
```

> **Note:** Ensure that you have [Conda](https://docs.conda.io/en/latest/) installed on your system.

## Manual Annotation

To perform manual annotation, follow these steps:

1. **Enter the 'MannalAnnotateInterface' folder**

2. **Start the Annotator:**
   - Execute the following command in your terminal:
   
     ```bash
     python Apollo_annotator.py
     ```

3. **Annotator Interface Overview:**
   - **Open Data:**  
     Click the **Open Data** button or press the shortcut **`O`** to load your data file. The file will be displayed in the main plot area.
   - **Load Windows:**  
     Click the **Load Windows** button or press **`L`** to load previously manually or automatically annotated windows.
   - **Select Window:**  
     Click the **Select Window** button or press **`A`** to define analysis windows by marking start and end points on the plot. Exit selection mode using **Stop Choosing** or press **`S`**.
   - **Refine Selections:**
     - **Redo:** Press the **Redo** button or **`R`** to remove the last window.
     - **Edit Window:** Use the **Edit Window** button to adjust window boundaries.
     - **Delete Window:** Use the **Delete Window** button to remove unwanted windows.
   - **Merge Windows:**  
     Click the **Merge Window** button or press **`M`** to combine overlapping windows.
   - **Update FFT:**  
     After setting your windows, click **Update FFT** to compute and display the Fast Fourier Transform results below the main plot. You can also configure the frequency and period range for visualization in the interface.
   - **Save Windows:**  
     Click the **Save Windows** button to export your window selections to a CSV file. The file will contain two columns: the first lists the start time and the second lists the end time for each annotated window.

## Automated Workflow

For an automated annotation workflow:

1. **Enter the 'AutoProcessWorkflow' folder**

2. **Configuration:**
   - Modify the target data path and the save data path in the configuration file 'disturbance_detection.yaml' under the folder.
   
3. **Execution:**
   - Run the script 'detect_example_script.py', which will automatically perform detection and save the results to the specified location.
