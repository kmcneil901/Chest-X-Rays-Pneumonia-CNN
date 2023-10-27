# Pneumonia Detection on Chest X-Ray images using Deep Learning 
### Authors: Kacey Clougher, Rachel Goldstein, Irwin Lam, and Kendall McNeil

Pneumonia is an infection that affects one or both lungs by causing the air sacs, or alveoli, of the lungs to fill up with fluid or pus. Traditionally, pneumonia detection hinges on the examination of chest X-ray radiographs, a labor-intensive process conducted by highly skilled specialists. This method often results in discordant interpretations among radiologists. Leveraging the power of deep learning techniques (CNNs), we have developed a computational approach for the detection of pneumonia regions.

Within this repository, you will find a comprehensive analysis that classifies x-ray images into two categories: Normal and Pneumonia. 
 
![4_SVH_Lung_Health_Pneumonia_final_1080p](https://github.com/kmcneil901/Chest-X-Rays-Pneumonia/assets/137820049/59a71e34-f3ef-40c6-8f01-d360931e1695)


## Repository Structure
- Preprocess and Final model notebook
- README.md: High-level README for reviewers of this project
- Technical Presentation

## Approach
We used Convolutional Neural Network (CNN) techniques to develop an AI system for pneumonia detection. The neural network model or architecture was designed using the Keras API and was implemented using Python and TensorFlow. 

### Data and Preprocessing
The dataset was provided by Kaggle: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/). It contains X-ray three folders/directories (train, test, val) of images divided into two categories: "Normal" and "Pneumonia." These images were acquired from pediatric patients aged one to five years at Guangzhou Women and Children's Medical Center. Before inclusion, a quality control process removed poor-quality scans, and expert physicians graded the diagnoses to prepare the dataset for AI training.

We consistently reshaped each directory in the dataset, followed by additional reshaping and transposing of individual datasets. Additionally, the data was normalized with a rescale factor of 1.0/255, and all images were resized to 64x64 dimensions. Given the relatively small size of the original challenge dataset, the image augmentations proved advantageous in mitigating overfitting. **Add we also add more data if we rotated it or used greyscale** 

### Key Tools
* Python
* TensorFlow
* Keras
* Numpy

### Modeling
**Add information about Irwin's function if included in final model**

## Reproducing the Experiment
Ensure your environment supports the most recent version of [Tensorflow](https://github.com/tensorflow/tensorflow/releases). If your environment does not support Tensorflow, we suggest using Google Colab for this model. **Please refer to the FINAL NOTEBOOK for the entire modeling process.**

## Presentation and Resources
 - You can view the technical presentation from the repository [here](https://github.com/kmcneil901/Chest-X-Rays-Pneumonia/blob/main/Pnuemonia_Classification_Model_Presentation.pdf).
 - The dataset was verified and provided by Kaggle: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/)
 - This project and approach drew inspiration from the article found here: [Efficient Pneumonia Detection in Chest Xray Images Using Deep Transfer Learning](https://www.mdpi.com/2075-4418/10/6/417)
