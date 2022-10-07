

# automatic-image-captioning
The attention mechanism gives more weight to certain regions of the image than others.This means that the central region of the image will be given far more importance than 
the surrounding region during captioning. The other regions in the image. So, the central region in the image is identified correctly in any case.

The layers have been added to the model in the following pattern
A CNN model used for the image vector.(Simple use of dense functions)
An LSTM RNN model used for the captions. (LSTM() function used that maintains memory).
The model has two inputs.
The CNN and RNN are combined using add() function.
The RNN based attention layers are added.
Tanh activation used, tanh is specifically used for the attention layers
