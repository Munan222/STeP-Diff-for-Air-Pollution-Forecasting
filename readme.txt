## Project Structure  

### Dataset Preparation  
- **`data_read.py`**: Handles data preprocessing.  
- **`/Data_sample/.`**: Preprocessed data samples.

### STeP-Diff's  main architecture
- **Training & Testing**: `exe_air_forecasting.py` – Executes model training and evaluation.  
- **Model Definition**: `main_model.py`, `air_forecasting.py` – Define the core architecture of STeP-Diff.  
- **Metric Computation**: `utils.py` – Computes evaluation metrics.  

### STeP-Diff's model Components
- **DeepONet**: `/DeepONet/.`
- **PDE**: `/PDE/.`

### Results Analysis  
- **`results.py`**: Analyzes and visualizes experimental results.  

## Usage  

### Step 1: Data Preparation  
Run `data_read.py` to preprocess the dataset. The input data should be formatted as a NumPy array of shape **[L, X, Y]**, where:  
- **L** represents the temporal dimension.  
- **X, Y** correspond to geographical coordinates (longitude and latitude).  

### Step 2: Model Training & Testing  
Execute `exe_air_forecasting.py` to train and evaluate the model. The optimal model parameters will be saved in the `save/` directory.  

### Step 3: Results Analysis  
Run `results.py` to further analyze and visualize the experimental results.  
