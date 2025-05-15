## 📁 Project Structure

### 🔧 Dataset Preparation  
- **`data_read.py`** – Handles dataset preprocessing and formatting.  
- **`/Data_sample/`** – Contains example preprocessed data samples.

### 🧠 STeP-Diff Architecture  
- **Training & Testing**:  
  - `exe_air_forecasting.py` – Executes the full training and evaluation pipeline.  
- **Model Definition**:  
  - `main_model.py`, `air_forecasting.py` – Define the core STeP-Diff architecture.  
- **Metric Computation**:  
  - `utils.py` – Computes evaluation metrics such as MAE, RMSE, etc.

### 🧩 Model Components  
- **DeepONet**:  
  - Implementation located in `/DeepONet/`  
- **PDE Module**:  
  - Custom physical constraint layers in `/PDE/`

### 📊 Results Analysis  
- **`results.py`** – Analyzes and visualizes the experimental outcomes.

---

## 🚀 Usage

### Step 1: Data Preparation  
Run the preprocessing script:

```bash
python data_read.py
