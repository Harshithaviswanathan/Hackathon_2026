# 🧠 Build → Break → Improve: Synthetic Image Detector

### 📌 Overview
This project implements a complete pipeline for detecting AI-generated images and evaluating model robustness.  
The goal is to simulate a real-world AI security workflow:
  - **Build** a detector to classify images as REAL or FAKE
  - **Break** the detector using adversarial modifications
  - **Improve** the system by analyzing vulnerabilities and proposing defenses
🔗 **Delpoyed link:** https://shamanistic-deirdre-noncalculably.ngrok-free.dev

### 🗂️ Dataset
**CIFAKE** — 120k images (Real vs Synthetic, 32×32 RGB)

### 📊 Methodology
- **Base Model:** ResNet-18 (binary classifier)
- **Metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix
- **Explainability:** Grad-CAM / saliency maps

### 🛡️ Adversarial Experiments
- Gaussian noise perturbations  
- Gaussian blur (artifact suppression)  
- FGSM attack  
These experiments show the detector’s sensitivity to high-frequency artifacts.

### 🚀 Proposed Improvement
Adversarial training with frequency-aware augmentations to improve robustness.

### 🛠️ Tech Stack
Python, PyTorch, Torchvision, NumPy, Matplotlib, Scikit-learn
