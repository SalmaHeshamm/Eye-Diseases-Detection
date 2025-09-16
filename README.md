# 👀 Eye Disease Classifier

A **Streamlit web app** to classify **3 types of eye diseases** using an **EfficientNet-B3 model**. Users can upload an eye image and instantly get the predicted disease along with the confidence score.

---

## Dataset

The dataset consists of **retinal images** for the following classes:

- **Normal**  
- **Diabetic Retinopathy**  
- **Cataract**  
- **Glaucoma**

---

## Features

- **Upload & preview images** in real-time.  
- **Predict 4 eye diseases**: Normal, Diabetic Retinopathy, Cataract, Glaucoma.  
- **Confidence score display** for the prediction.  
- **Fast & efficient model loading** using Streamlit caching.  

---

## Demo

![App Screenshot](images/Screenshot 2025-09-16 at 4.24.29 PM.png)  
*Screenshot showing the upload box and prediction result.*

![Prediction Result](images/Screenshot 2025-09-16 at 4.25.23 PM.png)  


---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/eye-disease-classifier.git
cd eye-disease-classifier
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run main.py
```
