# Nut-Detector
Detect three class of nuts kept in a colored tray under variable lighting and variable tray pose and distances on the stable frame of the video. The complete report is given [here](https://github.com/njanirudh/Nut-Detector/blob/master/report/CV19_project_paper.pdf) and the final presentation is [here](https://github.com/njanirudh/Nut-Detector/blob/master/report/cv19_FINAL_presentation.pdf).

---
### Introduction
The main goal of this task is to detect three classes of nuts ie. Haselnuts, Peanuts, Walnuts that are thrown randomly on a uniformly colored tray using computer vision. The image of the tray is taken at different light brightness and distances.
Extra objects called as distractors are used to confuse the object detector. 

<img src="/images/dataset.jpg" width="800"></img>


### Pipeline

The general pipeline consists of 3 steps. The stable frame detection is performed using classical computer vision method. The main object detection part involves training a deep learning model.

<img src="/images/Camera.png" width="800"></img>
