## TensorDual-VQC: Operator Learning for Scalable and Noise-Robust Variational Quantum Circuits

### Installation 

The main dependencies include *pytorch* and *torchquantum*

#### Torch Quantum 
```
pip3 install torchquantum
```

 ### 0. Downloading the dataset 
 #### The quantum dot dataset used in our experiments is available through an external repository
```
git clone https://gitlab.com/QMAI/mlqe_2023_edx.git
```

### 1. Simulating TensorDual-VQC experiments

#### 1.1 Quantum Dot Classification
python3 TensorDual_QD.py

#### 1.2 Max-Cut Problem 
python3 TensorDual_QAOA.py 

### 1.3 LiH Molecular Simulation
python3 TensorDual_QSim.py
