# Detect Znak by IIR

Project: Sign Detection with Nomeroff-Net

### Authors:

#### Bespalov Sergey

#### Fedotov Dmitriy

#### Milaev Denis

#### Pimonov Antom

##### Institution: Novosibirsk State University (NSU)

# Description:

This project aims to develop a system for detecting traffic signs using the Nomeroff-Net fine-tuned network. The system is currently under development, with a focus on improving the accuracy of text recognition.

# Getting Started:

To run the project, clone the repository and install the required dependencies:

git clone https://github.com/your-username/sign-detection.git
cd sign-detection
pip install -r requirements.txt
Inference:

To perform inference on an image, run the following command:

python inference.py --image_path <path_to_image>
This will output the detected signs and their corresponding bounding boxes.

#### Training:

To train the model, run the following command:

python train.py --data_dir <path_to_data_directory> --epochs <number_of_epochs>
This will train the model on the specified dataset and save the trained model to a file.

# Results:

![](public/6fc09578-c395-4339-b808-955c5983f3e7.jpg)
![](public/b53582f8-af8f-420b-a781-049a50dabc78.jpg)

Future Work:

The future work for this project includes:

Improving the accuracy of text recognition
Detecting additional types of traffic signs
Developing a real-time system for sign detection

### References:

Nomeroff-Net: https://github.com/ria-com/nomeroff-net