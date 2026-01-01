# Credit Card Fraud Detection Using Quantum Support Vector Machines



## 📖 Overview

This project evaluates the effectiveness of **Quantum Support Vector Machines (QSVM)** for credit card fraud detection under extreme class imbalance. Using PCA-reduced data and qubit configurations ranging from 4 to 10 qubits, we benchmark several quantum feature maps against classical SVM baselines.

### Key Findings

Our best-performing 10-qubit EfficientSU2 model achieves:
- **83.3% accuracy**
- **F1-score of 0.7368**
- **Superior performance** over classical baselines

These results demonstrate QSVM's ability to capture complex non-linear relationships in highly imbalanced transactional data.

### Why Quantum Machine Learning?

Classical ML models face critical challenges in fraud detection:

| Challenge | Impact |
|-----------|--------|
| **Extreme Class Imbalance** | Only 0.17% fraudulent transactions (492 out of 284,807) |
| **High Dimensionality** | 30 features lead to expensive training |
| **Evolving Fraud Patterns** | Models require constant retraining |
| **Non-linear Relationships** | Complex patterns difficult to capture |

**Quantum kernels** address these by mapping data into high-dimensional Hilbert spaces, enabling better separation of fraud and legitimate transactions.

## ✨ Features

- 🎯 **Multiple Quantum Feature Maps**: ZZ, Pauli, EfficientSU2, Custom Dense, High Entangling
- 📊 **Scalable Qubit Configurations**: 4, 6, 8, and 10 qubit implementations
- 🔄 **Hybrid Quantum-Classical Approach**: Combining quantum kernels with classical SVM
- 📈 **Comprehensive Benchmarking**: Against classical SVM with RBF, Polynomial, and Linear kernels
- ⚖️ **Advanced Preprocessing**: SMOTE, RandomUnderSampler, and PCA for handling imbalance
- 🚀 **GPU-Accelerated Simulation**: Efficient quantum circuit simulation

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Classical Preprocessing                   │
│  • RobustScaler (handle outliers)                           │
│  • PCA (30 features → 4/6/8/10 dimensions)                  │
│  • SMOTE + RandomUnderSampler (balance classes)             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Quantum Feature Encoding                        │
│                                                              │
│  Classical Data (x) → Quantum State |φ(x)⟩                 │
│                                                              │
│  Feature Maps:                                               │
│  • ZZ Feature Map (Z-axis entanglement)                     │
│  • Pauli Feature Map (X, Y, Z rotations)                    │
│  • EfficientSU2 (SU(2) + CX gates) ⭐ Best                 │
│  • Custom Dense (Rx, CX, Rz rotations)                      │
│  • High Entangling (Hadamard + Ry + CX)                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Quantum Kernel Computation                      │
│                                                              │
│  K(x, y) = |⟨φ(x)|φ(y)⟩|²                                  │
│                                                              │
│  Compute kernel matrices:                                    │
│  • K_train: similarities between training samples           │
│  • K_test: similarities between test and training samples   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Classification (SVM)                            │
│                                                              │
│  Two Approaches:                                             │
│  1. Pure QSVM: Quantum kernel → SVM classifier              │
│  2. Hybrid: Quantum kernel → Classical RBF SVM              │
│                                                              │
│  Output: Fraud (1) or Legitimate (0)                        │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) GPU for faster simulation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/QSVM-Credit-Card-Fraud-Detection.git
cd QSVM-Credit-Card-Fraud-Detection
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv qml_env
source qml_env/bin/activate  # On Windows: qml_env\Scripts\activate

# Or using conda
conda create -n qml_env python=3.8
conda activate qml_env
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
qiskit>=0.45.0
qiskit-aer>=0.13.0
qiskit-machine-learning>=0.7.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
imbalanced-learn>=0.11.0
tensorflow>=2.13.0
jupyter>=1.0.0
```

### 4. Download Dataset

The TensorFlow Credit Card Fraud Dataset will be automatically downloaded on first run, or manually:

```bash
python scripts/download_data.py
```

## 🚀 Quick Start

### 1. Run Complete Pipeline

```bash
python main.py --qubits 10 --feature_map efficientsu2
```

### 2. Train Classical Baseline

```bash
python train_classical.py --kernel rbf
```

### 3. Train Quantum Models

```bash
# Single feature map
python train_quantum.py --qubits 8 --feature_map zz

# All feature maps
python train_quantum.py --qubits 10 --all_maps
```

### 4. Hybrid Approach

```bash
python train_hybrid.py --qubits 10 --feature_map efficientsu2
```

### 5. Run Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

Available notebooks:
- `01_data_exploration.ipynb` - Dataset analysis and visualization
- `02_classical_baseline.ipynb` - Classical SVM experiments
- `03_quantum_feature_maps.ipynb` - Quantum kernel evaluation
- `04_results_analysis.ipynb` - Performance comparison and visualization

## 📊 Results

### Performance Summary

| Model Type | Best Configuration | Accuracy | F1-Score | AUC |
|------------|-------------------|----------|----------|-----|
| **Classical SVM** | RBF Kernel (8-qubit PCA) | 80.0% | 0.5714 | 0.6807 |
| **Quantum SVM** | EfficientSU2 (10 qubits) | **83.3%** | **0.7368** | **0.7189** |
| **Hybrid** | EfficientSU2 + RBF | **83.3%** | 0.7100 | **0.7224** |

### Feature Map Comparison

| Feature Map | Qubits | Accuracy | F1-Score | Recall |
|-------------|--------|----------|----------|--------|
| **EfficientSU2** ⭐ | 10 | **83.3%** | **0.7368** | 93.3% |
| Custom Dense | 10 | 81.7% | 0.6800 | 86.7% |
| High Entangling | 10 | 81.7% | 0.6800 | 86.7% |
| ZZ Feature Map | 10 | 78.3% | 0.6190 | 80.0% |
| Pauli Feature Map | 10 | 75.0% | 0.5455 | 73.3% |
| Classical RBF | 8 | 80.0% | 0.5714 | 66.7% |

### Key Insights

1. **Qubit Count Matters**: Performance consistently improves from 4 → 6 → 8 → 10 qubits
2. **Entanglement Depth**: Deep entanglement (EfficientSU2, Custom Dense) outperforms shallow circuits
3. **Quantum Advantage**: Quantum kernels excel at capturing non-linear fraud patterns
4. **Hybrid Approach**: Achieves highest AUC by combining quantum embeddings with classical learning

### Confusion Matrix - Best Model

```
              Predicted
              Fraud  Normal
Actual Fraud    14      1      (93.3% Recall)
      Normal     9     36      (80.0% Specificity)

True Positives: 14 | False Positives: 9
False Negatives: 1 | True Negatives: 36
```


## 🔬 Theory

### Quantum Kernel Function

The quantum kernel measures similarity between data points in Hilbert space:

```
K(x, y) = |⟨φ(x)|φ(y)⟩|²
```

Where:
- `|φ(x)⟩ = U_φ(x)|0⟩^⊗n` is the quantum state embedding
- `U_φ(x)` is the parametrized quantum circuit (feature map)
- `n` is the number of qubits

### Classical SVM Optimization

The SVM finds the maximum-margin hyperplane:

```
min_(w,b,ξ) ½||w||² + C Σ ξ_i

subject to:
  y_i(w^T x_i + b) ≥ 1 - ξ_i
  ξ_i ≥ 0
```

### Kernel Matrices

For training and testing:

```
K_train[i,j] = K(x_i, x_j)  where x_i, x_j ∈ X_train
K_test[i,j] = K(x_i, x_j)   where x_i ∈ X_test, x_j ∈ X_train
```

### Feature Maps Overview

#### 1. **ZZ Feature Map**
- Z-axis rotations with entanglement
- Captures pairwise feature interactions
- Moderate depth circuits

#### 2. **Pauli Feature Map**
- Rotations along X, Y, Z axes
- Basic entanglement patterns
- Limited expressiveness

#### 3. **EfficientSU2 Feature Map** ⭐
- Layers of SU(2) rotations (Rx, Ry, Rz)
- CX entangling gates
- Highly expressive for non-linear patterns
- **Best performer in experiments**

#### 4. **Custom Dense Feature Map**
- Rx → CX → Rz rotation sequence
- Deep multi-qubit correlations
- Strong performance on fraud detection

#### 5. **High Entangling Feature Map**
- Hadamard initialization (superposition)
- Ry rotations + CX entanglement
- Captures complex feature relationships

## 📈 Performance Visualizations

### 1. Average AUC by Model Type
- **Hybrid Models**: 0.7224 (highest)
- **Quantum Models**: 0.7189
- **Classical Models**: 0.6807

### 2. Performance by Qubit Count
- **4 Qubits**: Baseline performance
- **6 Qubits**: Marginal improvement
- **8 Qubits**: Significant gains
- **10 Qubits**: Best results (83.3% accuracy)

### 3. Feature Map Rankings
1. EfficientSU2 (F1: 0.7368)
2. Custom Dense (F1: 0.6800)
3. High Entangling (F1: 0.6800)
4. ZZ (F1: 0.6190)
5. Pauli (F1: 0.5455)

## 🚀 Future Work

- [ ] **Real Quantum Hardware**: Test on IBM Quantum or other quantum processors
- [ ] **Quantum Feature Selection**: Use quantum algorithms to identify optimal features
- [ ] **Deeper Circuits**: Explore 12-16 qubit configurations
- [ ] **Error Mitigation**: Implement noise-resilient techniques for real devices
- [ ] **Multi-Class Detection**: Extend to different fraud types
- [ ] **Real-Time Deployment**: Optimize for production fraud detection systems
- [ ] **Federated Quantum Learning**: Privacy-preserving distributed training
- [ ] **Quantum Autoencoders**: Unsupervised anomaly detection

## 🤝 Contributing

We welcome contributions! Areas for improvement:

1. **New Feature Maps**: Design novel quantum circuits for fraud detection
2. **Optimization**: Improve quantum circuit efficiency
3. **Benchmarking**: Add comparisons with other QML approaches
4. **Documentation**: Enhance code documentation and tutorials
5. **Testing**: Expand test coverage



## 👥 Authors

**LUMS SBASSE Quantum Machine Learning Team**

| Name | Role |
|------|------|
| Rida Arshad | Research & Implementation |
| Nawal Shahid | Quantum Circuit Design |
| Talha Nasir | Experimentation & Analysis |

**Supervisor**: Dr. Muhammad Faryad

**Institution**: Syed Babar Ali School of Science and Engineering (SBASSE),  
Lahore University of Management Sciences (LUMS)

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@techreport{arshad2024qsvm,
  title={Credit Card Fraud Detection Using Quantum Support Vector Machines},
  author={Arshad, Rida and Shahid, Nawal and Nasir, Talha},
  institution={Lahore University of Management Sciences},
  year={2024},
  type={Technical Report}
}
```

## 📖 References

1. **Grossi, M., et al.** (2023). "Mixed Quantum-Classical Method for Fraud Detection with Quantum Feature Selection." [arXiv:2208.07963](https://arxiv.org/abs/2208.07963)

2. **Micheal, L., et al.** (2024). "Evaluating the Efficacy of Quantum Support Vector Machines in Detecting Synthetic Identity Fraud in Financial Datasets."

3. **Cortés, C., & Vapnik, V.** (1995). "Support-Vector Networks." *Machine Learning*, 20(3), 273-297.

4. **Yin, T.** (2024). "Quantum support vector machines: theory and applications." *CONF-MPCS 2024 Workshop*.

5. **Schnabel, J., & Roth, M.** (2024). "Quantum Kernel Methods under Scrutiny: A Benchmarking Study." [arXiv:2409.04406](https://arxiv.org/abs/2409.04406)

6. **Singh, N., & Pokhrel, S.R.** (2025). "Modeling Feature Maps for Quantum Machine Learning." [arXiv:2501.08205](https://arxiv.org/abs/2501.08205)

## 🙏 Acknowledgments

Special thanks to:

- **IBM Quantum** for providing quantum computing resources
- **Qiskit Community** for the quantum machine learning framework
- **TensorFlow** for the credit card fraud dataset
- **LUMS SBASSE** for computational resources and support



---

<div align="center">

</div>
