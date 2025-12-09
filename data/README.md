# Data
The data used to train the model is the "speech_commands" dataset from Tensorflow, provided here: [https://www.tensorflow.org/datasets/catalog/speech_commands](https://www.tensorflow.org/datasets/catalog/speech_commands).

The two files listed in this directory are 1) a list of ten key words representing 'off' and 'yes' used to verify the accuracy of the hardware implementation, and 2) a trace file collected from the communication of the two chips. This is used to determine the output accuracy of the hardware, and diagnose any communication issues.
