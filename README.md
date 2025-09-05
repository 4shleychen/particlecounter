# 🔬 Particle Counter App

A user-friendly web application that uses object detection to analyze images of microscopic objects — such as microcarriers or cells. Simply upload an image, set your scale and size constraints, and receive a full quantitative analysis, including count, size distribution, and visualization.

## ✨ Features

* 📷 **Image Upload**: Drag-and-drop or browse to upload your microscopy images.
* ⚙️ **Custom Parameters**:

  * Set image **scale** (e.g. µm/pixel)
  * Define **minimum and maximum diameters**
* 🔍 **Object Detection**:

  * Automatically detects circular/elliptical particles (e.g. microcarriers, cells)
* 📊 **Results Provided**:

  * Particle count
  * Min / Max / Average diameter
  * Size distribution histogram
  * Tabular data export (CSV)
* 📈 **Visual Output**:

  * Annotated detection overlay
  * Interactive histogram
  * Copy-able results

---

## 🎥 Video Demo

WATCH HOW THE APP WORKS IN ACTION:
https://github.com/user-attachments/assets/88114bc9-b71c-404f-b388-41644bf3ec19

---

## 🧠 How It Works

1. Uploaded image is processed, resized and converted to grayscale.
2. Blob detection / contour finding is performed using OpenCV.
3. Scale and size thresholds are used to filter out noise or irrelevant particles.
4. Statistics are calculated and returned to the user.

---

## 📂 Output Files

* Annotated image with bounding circles
* Variety of data which can be copied to clipboard!

---

## 📌 Use Cases

* Microcarrier growth monitoring
* Cell culture analysis
* Particle sizing
* Quality control in biomedical manufacturing

---

## 🛠️ Future Improvements

* Batch processing support
* User accounts and saved analysis
* Integration with microscopy hardware

---

## 📬 Contact

For feature requests, bug reports, or collaborations:
📧 [ashley.chen.2302@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/ashley-chen1/)
