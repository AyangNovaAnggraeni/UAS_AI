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

- It contains 3 source code with py  extension, and the image files are a result of the training mode.
- For the dataset :
  - Sanjaya, Samuel Ady (2024), “BISINDO Indonesian Sign Language: Alphabet Image Data”, Mendeley Data, V1, doi: 10.17632/ywnjpbcz8m.1
  - https://www.kaggle.com/datasets/achmadnoer/alfabet-bisindo
- And the pdf file is a draft journal for further explanation


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
This project was created as a group project for the Artificial Intelligence course.
Developed by:

Ayang Nova Anggraeni (me)

Julia Ayu Dewi Siagian (my teammate)

We collaborated on building the dataset, training the model, writing code, and preparing the final report.

## 📝 Disclaimer

This repository is created for **learning purposes only**.  
It contains my personal work for the Final Project in "Kecerdasan Buatan" class in Universitas Singaperbangsa Karawang.
I do not claim that this is the best way to structure or solve the problem — these files are meant to serve as **references only**, not as official guides.
Please feel free to learn from the code, but use your own judgment and consult other sources when applying it in real projects.

---
