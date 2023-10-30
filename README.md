# Pneumonia Detection in Chest X-Rays - Convolutional Neural Network
### Kacey Clougher, Rachel Goldstein, Irwin Lam, and Kendall McNeil

Pneumonia is an infection that affects one or both lungs by causing the air sacs, or alveoli, of the lungs to fill up with fluid or pus. Traditionally, pneumonia detection hinges on the examination of chest X-ray radiographs, a labor-intensive process conducted by highly skilled specialists. This method often results in discordant interpretations among radiologists. Leveraging the power of deep learning techniques (convolutional neural networks), we have developed a computational approach for the detection of pneumonia regions.

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

We employed the ImageDataGenerator to augment our dataset, enabling diverse data transformations, such as scaling, rotation, flipping, zooming, and shifting. Additionally, the data was normalized with a rescale factor of 1.0/255, and all images were resized to 64x64 dimensions. These transformations generated modified versions of the original images, effectively increasing our training data. This was particularly beneficial for our relatively small dataset of approximately 6000 images.

### Key Tools
* Python
* TensorFlow
* Keras
* Numpy

### Modeling
This model architecture consists of an Input Layer with a dropout for overfitting prevention, a First Hidden Layer with 32 convolutional filters (3x3 kernel) and tanh activation, a Second Hidden Layer with max-pooling (2x2) and another convolutional layer with 64 filters and tanh activation, and a Third Hidden Layer with additional max-pooling and a final convolutional layer with 64 filters and tanh activation. Dropout layers are used after each hidden layer to further mitigate overfitting. In summary, this model is a convolutional neural network with dropout layers, max-pooling layers, and a final dense layer for binary classification. 

![model (1)](https://github.com/kmcneil901/Chest-X-Rays-Pneumonia/assets/137820049/948a95fc-8561-404c-bd73-5b0824dfcb7a)  

## Reproducing the Experiment
Ensure your environment supports the most recent version of [Tensorflow](https://github.com/tensorflow/tensorflow/releases). If your environment does not support Tensorflow, we suggest using Google Colab for this model. **Please refer to the FINAL NOTEBOOK for the entire modeling process.**

## Presentation and Resources
 - You can view the technical presentation from the repository [here](https://github.com/kmcneil901/Chest-X-Rays-Pneumonia/blob/main/Pnuemonia_Classification_Model_Presentation.pdf).
 - The dataset was verified and provided by Kaggle: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/)
 - This project and approach drew inspiration from the article found here: [Efficient Pneumonia Detection in Chest Xray Images Using Deep Transfer Learning](https://www.mdpi.com/2075-4418/10/6/417)
