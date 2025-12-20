
# 🧠 Hybrid Multi-Modal Deep Learning for Alzheimer's Disease Detection  
## Using MRI Images and Clinical Metadata

<p align="center">
  <img src="https://raw.githubusercontent.com/aniketghoshxdaksh-pixel/Alzheimers-Detection/main/Model%20Architecture.png" alt="Model Architecture" width="900"/>
</p>

**Author:** Aniket Ghosh  
**Institution:** Department of Computer Science and Engineering, National Institute of Technology Calicut  
**Email:** aniketm240303cs@nitc.ac.in  

---

### 🚀 Project Overview

This repository contains a **hybrid multi-modal deep learning model** for early and accurate 4-class classification of Alzheimer's Disease using the OASIS dataset.

- **CNN Branch** → Extracts structural features from 2D axial MRI slices (brain atrophy patterns)
- **MLP Branch** → Processes 7 clinical metadata features: `Age`, `Educ`, `SES`, `MMSE`, `eTIV`, `nWBV`, `ASF`
- **Feature Fusion** → Concatenation → Final softmax classifier

**Classes**: NonDemented • VeryMildDemented • MildDemented • ModerateDemented

**Key Achievements**:
- Test Accuracy: **~84%**
- NonDemented F1-score: **0.93**
- MildDemented Recall: **0.80** (highly valuable for early screening)

---

### 📊 Model Performance: Final Confusion Matrix

<p align="center">
  <img src="https://raw.githubusercontent.com/aniketghoshxdaksh-pixel/Alzheimers-Detection/main/confusion%20matrix.png" alt="Confusion Matrix" width="800"/>
</p>

*Excellent classification of NonDemented cases (38,532 correct). Good true positives for MildDemented (523). Some misclassification in VeryMildDemented due to severe class imbalance.*

---

### 🖼️ Sample MRI Slices – Disease Progression Visualization

<p align="center">
  <img src="https://raw.githubusercontent.com/aniketghoshxdaksh-pixel/Alzheimers-Detection/main/MRI.png" alt="MRI Sample Grid" width="900"/>
</p>

#### Individual Class Examples

<p align="center">
  <img src="https://raw.githubusercontent.com/aniketghoshxdaksh-pixel/Alzheimers-Detection/main/NotDemented.png" alt="NonDemented" width="22%"/> &nbsp;
  <img src="https://raw.githubusercontent.com/aniketghoshxdaksh-pixel/Alzheimers-Detection/main/VeryMildDemented.png" alt="VeryMildDemented" width="22%"/> &nbsp;
  <img src="https://raw.githubusercontent.com/aniketghoshxdaksh-pixel/Alzheimers-Detection/main/MildDemented.png" alt="MildDemented" width="22%"/> &nbsp;
  <img src="https://raw.githubusercontent.com/aniketghoshxdaksh-pixel/Alzheimers-Detection/main/ModeratedDemented.png" alt="ModerateDemented" width="22%"/>
</p>

*Visual signs of progression: enlarged ventricles, widened sulci, and hippocampal atrophy increase with severity.*

---

### 🛠 Technologies & Frameworks Used

<p align="center">
  <img src="https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/stream_executor/platform/default/dso_loader.cc" alt="TensorFlow" height="60"/> <!-- Placeholder; actual logo URLs below -->
  <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="TensorFlow" height="60"/> &nbsp;&nbsp;
  <img src="https://www.vectorlogo.zone/logos/keras/keras-icon.svg" alt="Keras" height="60"/> &nbsp;&nbsp;
  <img src="https://numpy.org/images/logo.svg" alt="NumPy" height="60"/> &nbsp;&nbsp;
  <img src="https://pandas.pydata.org/static/img/pandas_white.svg" alt="Pandas" height="60"/> &nbsp;&nbsp;
  <img src="https://matplotlib.org/stable/_images/logos2/matplotlib_128.png" alt="Matplotlib" height="60"/> &nbsp;&nbsp;
  <img src="https://seaborn.pydata.org/_images/logo-wide-lightbg.svg" alt="Seaborn" height="60"/> &nbsp;&nbsp;
  <img src="https://scikit-learn.org/stable/_static/scikit-learn-logo-without-background.svg" alt="scikit-learn" height="60"/> &nbsp;&nbsp;
  <img src="https://opencv.org/wp-content/uploads/2020/07/OpenCV_logo_white.png" alt="OpenCV" height="60"/> &nbsp;&nbsp;
  <img src="https://www.kaggle.com/static/images/site-logo.png" alt="Kaggle" height="50"/>
</p>

**Core Libraries**:
- TensorFlow / Keras (Deep Learning)
- NumPy, Pandas, OpenCV (Data Processing)
- Matplotlib, Seaborn (Visualization)
- scikit-learn (Metrics & Utilities)
- Executed on Kaggle GPU Notebook

---

### 📂 Repository Contents

- `Alzheimers-Detection.ipynb` → Complete pipeline (data loading, model, training, evaluation)
- `Report.pdf` → Full academic report (methodology, results, references)
- Image files → Architecture diagram, confusion matrix, MRI samples
- `README.md` → This file

---

### ⚙️ Installation & Requirements

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python tqdm
```

Run the notebook directly on GitHub, Colab, or Kaggle.

---

### 📜 License

MIT License – free to use, modify, and distribute.

---

<p align="center">
  ⭐ <strong>Star this repo if you find it helpful!</strong> ⭐<br>
  Issues, forks, and contributions are very welcome! 🚀
</p>

*Academic Project • NIT Calicut • December 2025*


