# 📚 Final Project AI: Real-Time Indonesian Sign Language (BISINDO) Recognition Using CNN

## 🎯 Project Overview

This project is a final assignment for the **Artificial Intelligence** course at Universitas Singaperbangs Karawang. It aims to build an AI system that can recognize **Indonesian Sign Language (BISINDO)** hand gestures using **Convolutional Neural Networks (CNN) / Computer Vision** and translate them into text in real-time using a webcam.

The system supports **fingerspelling** recognition for A–Z letters. Detected letters are buffered into words and sentences, which are displayed live along with their **confidence level**.

---

## 🧠 Key Features

* Real-time sign language recognition via webcam
* Letter-by-letter BISINDO gesture classification (A-Z)
* Dynamic Region of Interest (ROI)
* Threshold-based prediction filtering
* Buffered word & sentence builder
* Visualization of prediction confidence
* Optional voice output (Text-to-Speech)

---

## 📁 Project Structure

```

---



## 🚀 How to Run

### 1. Train the Model (if not using pre-trained)

```bash
python train_model.py
```

### 2. Run Real-Time Prediction (Webcam)

```bash
python predict_live.py
```

### 3. Predict from an Image

```bash
python predict_recorded.py
```

---

## 📝 Notes

* Dataset used is a custom Indonesian Sign Language (BISINDO) fingerspelling dataset.
* The model classifies one letter at a time.
* Sentence formation is based on timing and confidence of predictions.
* Letter buffering and word segmentation are handled manually (e.g. through delay and keypress events).
* This project is a demonstration of **supervised learning** using **CNN for image classification**.

---

## 🎓 Course Information

* **Course**: Kecerdasan Buatan
* **University**: Universitas Singaperbangsa Karawang
* **Semester**: 4 

---

## 🙋‍♀️ Author
This is a group project
* **Name**: 


---
