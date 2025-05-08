# 🔧 CNC Spindle Current Prediction using LSTM and Transfer Learning

This project explores an energy-efficient predictive maintenance strategy for CNC milling machines by using **Long Short-Term Memory (LSTM)** networks enhanced with **Transfer Learning**. It enables accurate spindle current prediction across varying materials and machining conditions with reduced labeled data and computational cost.

---

## 🚀 Overview

- Implements a transfer learning framework using LSTM networks for **time-series spindle current prediction**.
- Optimizes **energy usage and predictive maintenance** in CNC machines.
- Generalizes predictions across **different materials** (e.g., steel ↔ aluminum) and **machining setups** (e.g., with/without air cut, different component types).
- Achieves **high accuracy** with significantly fewer labeled samples in the target domain.
- Trained and validated on real-world CNC datasets collected from various operational scenarios.

---

## 🧠 Key Contributions

- ✅ **Transfer learning-enabled LSTM** model for scalable spindle current prediction.
- ✅ **Reduced data dependency** through knowledge transfer from one domain to another.
- ✅ Robust generalization across **materials**, **machining methods**, and **geometries**.
- ✅ Data-efficient fine-tuning and optimization using early stopping, dropout, and learning rate scheduling.
- ✅ Thorough evaluation using **MSE, RMSE, MAE, R²**.

---

## 📁 Dataset

- Collected from CNC milling of objects called *Bautiel 1* and *Bautiel 2* under varying conditions.
- Includes measurements for spindle current, tool position, velocities, accelerations, and air-cut status.
- 8 variations captured: 2 components × 2 materials (Steel/Aluminum) × With/Without air cut.
- Spindle current (`curr_sp`) is the target variable.

---

## ⚙️ Model Architecture

- 2-layer LSTM with dropout and ReLU activations.
- Input reshaped into sequences of 10 time steps for temporal modeling.
- Final dense layer outputs spindle current (regression).
- Optimizer: Adam | Loss: MSE | Metrics: RMSE, MAE, R²
- Early stopping and checkpointing used during both base training and transfer learning.

---

## 🔄 Transfer Learning Strategy

1. **Base Training**: Train LSTM on one material/component/condition (e.g., Steel + Bautiel 1 without air cut).
2. **Fine-Tuning**: Unfreeze last 2 layers, and adapt to new data (e.g., Aluminum or Bautiel 2).
3. **Optimization**: Fine-tune for 50 epochs with reduced learning rate (0.0001).

**Transfer cases include:**
- 🔁 Steel ↔ Aluminum (cross-material)
- 🔁 Bautiel 1 ↔ Bautiel 2 (cross-component)
- 🔁 With ↔ Without air-cut (cross-condition)

---

## 📊 Results Summary

| Transfer Setup | RMSE ↓ | MAE ↓ | R² ↑ |
|----------------|--------|-------|------|
| Steel → Aluminum (B1, no air-cut) | 0.386 | 0.301 | 0.959 |
| Aluminum → Steel (B1, no air-cut) | 1.139 | 0.796 | 0.968 |
| Bautiel 1 → Bautiel 2 (Steel)     | 0.746 | 0.321 | 0.694 |
| Bautiel 2 → Bautiel 1 (Steel)     | 0.684 | 0.421 | 0.872 |

- Steel-to-Aluminum transfer showed **strong generalization**.
- Transfer from simpler to more complex patterns (B1 → B2) was **more effective** than the reverse.
- **Air-cut presence** did not significantly affect transferability.

---

## 🔬 Evaluation Metrics

- **MSE**: Penalizes large deviations (sensitive to spikes).
- **RMSE**: Same units as target (e.g., spindle current).
- **MAE**: Average absolute deviation.
- **R²**: Proportion of variance explained.

---

## 📦 How to Use

```bash
# Clone and install dependencies
git clone https://github.com/your-username/cnc-lstm-transfer.git
cd cnc-lstm-transfer
pip install -r requirements.txt

# Train base model
python train_lstm.py --dataset data/steel_b1.csv

# Fine-tune on target domain
python transfer_train.py --base_model checkpoints/base_model.pth --target_data data/aluminum_b1.csv
```

---

## 💡 Future Work

- Explore **GRU, CNN-LSTM, or attention** for better generalization.
- Integrate **feature selection** (e.g., PCA, L1 regularization).
- Extend to more diverse **multi-material** datasets.
- Add **explainability** tools for industrial deployment.

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 👥 Authors

Ajay Raja Ram Alageshan, Bhavya Baburaj Chovatta Valappil,  
Dinesh Babu Gopinath Hemavathi, Sai Nithin Reddy Ankireddy, Yashika Balaji  
Institute for Intelligent Cooperating Systems, Otto von Guericke University Magdeburg  
Project under Autonomous Multisensor Systems (AMS)