# STM32 Binarized Neural Network (BNN)

Deploy Binarized Neural Networks (BNNs) AI on any Microcontroller Unit (MCU).

As an example, I deploy model inference on an STM32-F411CEU6 MCU, running at 100 MHz (0.1GHz). Using the MNIST dataset for numerical recognition classification tasks, the network utilizes an MLP architecture with a total of 7,890 parameters.

Each inference takes **ONLY 1.3 ms**, with an overall accuracy rate of 93.89%.

For training Binarized Neural Networks and automatically generating C code, please refer to the repository: <https://github.com/ittuann/Binarized-Neural-Networks>

![Running Screenshot](Images/Screenshot.png)
