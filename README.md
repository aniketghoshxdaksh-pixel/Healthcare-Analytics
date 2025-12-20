
# 🧠 Hybrid Multi-Modal Deep Learning for Alzheimer's Disease Detection  
## Using MRI Images and Clinical Metadata

<p align="center">
  <img src="https://raw.githubusercontent.com/aniketghoshxdaksh-pixel/Alzheimers-Detection/main/Model%20Architecture.png" alt="Model Architecture" width="900"/>
</p>

**Author:** Aniket Ghosh  
**Institution:** Department of Computer Science and Engineering, National Institute of Technology Calicut  
**Email:** aniketm240303cs@nitc.ac.in  

[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-View%20Notebook-blue?logo=kaggle)](https://www.kaggle.com/code/daksh4/hybrid-cnn-clinical-metadata-84)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aniketghoshxdaksh-pixel/Alzheimers-Detection/blob/main/Alzheimers-Detection.ipynb)  

---

### 🚀 Project Overview

This project implements a **hybrid multi-modal deep learning framework** for 4-class Alzheimer's Disease classification using the public OASIS dataset.

- **CNN Branch**: Extracts brain atrophy patterns from 2D axial MRI slices
- **MLP Branch**: Processes 7 clinical features (`Age`, `Educ`, `SES`, `MMSE`, `eTIV`, `nWBV`, `ASF`)
- **Feature Fusion**: Concatenation → Final classifier

**Key Results**:
- Test Accuracy: **~84%**
- NonDemented F1-score: **0.93**
- MildDemented Recall: **0.80** (critical for early detection)

---

### 📊 Model Performance: Final Confusion Matrix

<p align="center">
  <img src="https://raw.githubusercontent.com/aniketghoshxdaksh-pixel/Alzheimers-Detection/main/confusion%20matrix.png" alt="Confusion Matrix" width="800"/>
</p>

*Strong NonDemented classification (38,532 correct). Solid MildDemented detection (523 TP). Minor confusion in VeryMildDemented due to class imbalance.*

---

### 🖼️ Sample MRI Slices – Disease Progression

<p align="center">
  <img src="https://raw.githubusercontent.com/aniketghoshxdaksh-pixel/Alzheimers-Detection/main/MRI.png" alt="MRI Sample Grid" width="900"/>
</p>

#### Individual Examples

<p align="center">
  <img src="https://raw.githubusercontent.com/aniketghoshxdaksh-pixel/Alzheimers-Detection/main/NotDemented.png" alt="NonDemented" width="22%"/> &nbsp;
  <img src="https://raw.githubusercontent.com/aniketghoshxdaksh-pixel/Alzheimers-Detection/main/VeryMildDemented.png" alt="VeryMildDemented" width="22%"/> &nbsp;
  <img src="https://raw.githubusercontent.com/aniketghoshxdaksh-pixel/Alzheimers-Detection/main/MildDemented.png" alt="MildDemented" width="22%"/> &nbsp;
  <img src="https://raw.githubusercontent.com/aniketghoshxdaksh-pixel/Alzheimers-Detection/main/ModeratedDemented.png" alt="ModerateDemented" width="22%"/>
</p>

*Enlarged ventricles, widened sulci, and hippocampal atrophy increase with disease severity.*

---

### 📄 Project Report – First Page Preview

<p align="center">
  <img src="report_preview.png" alt="Report First Page" width="800"/>
</p>

[Download Full Report (PDF)](Report.pdf)
---

### 🛠 Technologies & Frameworks

<p align="center">
  <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="TensorFlow" height="60"/> &nbsp;
  <img src="https://www.vectorlogo.zone/logos/keras/keras-icon.svg" alt="Keras" height="60"/> &nbsp;
  <img src="https://numpy.org/images/logo.svg" alt="NumPy" height="60"/> &nbsp;
  <img src="https://pandas.pydata.org/static/img/pandas_white.svg" alt="Pandas" height="60"/> &nbsp;
  <img src="https://matplotlib.org/stable/_images/logos2/matplotlib_128.png" alt="Matplotlib" height="60"/> &nbsp;
  <img src="https://seaborn.pydata.org/_images/logo-wide-lightbg.svg" alt="Seaborn" height="60"/> &nbsp;
  <img src="https://scikit-learn.org/stable/_static/scikit-learn-logo-without-background.svg" alt="scikit-learn" height="60"/> &nbsp;
  <img src="https://opencv.org/wp-content/uploads/2020/07/OpenCV_logo_white.png" alt="OpenCV" height="60"/> &nbsp;
  <img src="https://www.kaggle.com/static/images/site-logo.png" alt="Kaggle" height="50"/>
</p>

**Stack**: TensorFlow • Keras • NumPy • Pandas • OpenCV • Matplotlib • Seaborn • scikit-learn

---

### 📂 Repository Contents

- `Alzheimers-Detection.ipynb` → Full training & evaluation pipeline
- `Report.pdf` → Complete academic report
- Image files → Architecture, confusion matrix, MRI samples, report preview
- `README.md` → This file

---

### ⚙️ Run the Project

- **Live on Kaggle** (GPU enabled): [View & Run Notebook](https://www.kaggle.com/code/daksh4/hybrid-cnn-clinical-metadata-84)
- **Open in Colab**: Click the badge above
- **Local**: Download notebook + install requirements

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python tqdm
```

---

### 📜 License

MIT License – free to use and modify.

---

<p align="center">
  ⭐ <strong>Star the repo if you like it!</strong> ⭐<br>
  Contributions & feedback welcome! 🚀
</p>

*Academic Project • NIT Calicut • December 2025*




