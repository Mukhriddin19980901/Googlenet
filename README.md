# Googlenet - Inception_model_1

<img src="https://github.com/Mukhriddin19980901/Googlenet/blob/main/inception_definition.png" width="1000" height="600" />


The code defines a custom **GoogleNet**-like model called **GoogleNet_2** using **TensorFlow/Keras**. This model is designed for ***image classification*** tasks and is structured using various **Inception** modules. It also includes two classifier branches (class_1 and class_2) for ***multi-branch classification***. Here's a breakdown of the code:

**1.Model Initialization**: The GoogleNet_2 class is initialized with the input shape and the number of output classes (classes). It also defines the name of the model.

**2.Model Architecture**:

<img src="https://github.com/Mukhriddin19980901/Googlenet/blob/main/googlnet1.png" width="1000" height="500" />


-The model starts with a series of convolutional, batch normalization, and ReLU layers to process the input data.

-It then includes max-pooling layers to downsample the spatial dimensions of the feature maps.

-The Inception modules (inception_layer) are used to capture features at different scales and complexities.

-After the Inception modules, there are classifier layers (classifier_layer) connected to different Inception stages (inception_4a and inception_4d).

-The model ends with additional layers, including batch normalization, ReLU activation, average pooling, dropout, and a dense (fully connected) layer for classification.

**3.Model Building**:

-The ***GoogleNet_2*** class builds three separate models:

-inception_model: The main model that produces the final output.



<img src='https://github.com/Mukhriddin19980901/Googlenet/blob/main/nception-module-of-GoogLeNet-This-figure-is-from-the-original-paper-10.png' width='600' height='600' />



-class_1: A classifier model connected to one of the Inception stages (inception_4a).

-class_2: A classifier model connected to another Inception stage (inception_4d).



<img src='https://github.com/Mukhriddin19980901/Googlenet/blob/main/xYZlf.png' width='600' height='600' />


<img src='https://github.com/Mukhriddin19980901/Googlenet/blob/main/Auxilary.png' width='1400' height='1400' />


-**call** Method: The call method specifies how data should flow through the model. It takes an input tensor x and passes it through the ***inception_model, class_1,*** and ***class_2***, returning the outputs of each.

4.**Model Outputs**: The model produces three outputs:

-The inception_model output, representing the final features.

-The class_1 output, which is a classification result based on one of the Inception stages.

-The class_2 output, which is another classification result based on a different Inception stage.

   Overall, this code defines a complex neural network architecture that combines features from different Inception stages and provides multiple classification results. Keep in mind that configuring and training this model will require specifying a loss function, optimizer, and dataset, as well as conducting the training process.

**5**.Because my model is **functional**, I've established training and testing sessions using a function decorated with **@tf.function**. This approach enables us to train the model effectively. Initially, we create instances of the **loss** 
function and **optimizer**.Our data labels are of type ***'uint16'***, so we've opted for the **"SparseCategoricalCrossEntropy"** loss function and the **"Adam"** optimizer.
   
**6**.I trained the model for 10 epochs to save time, but you have the flexibility to adjust the number of epochs to reduce loss and improve accuracy. During the initial phases of the training session, you'll notice a significant drop in both ***loss*** and ***validation loss***. However, as training progresses, the difference becomes less pronounced. On the other hand, training ***accuracy*** dropped a little bit at first but remained unchanged till the last with small flactuations.but test metrics steadily increase after the second **epoch**.

<img src="https://github.com/Mukhriddin19980901/Googlenet/blob/main/losses_inc.png" width="600" height="600" />

<img src="https://github.com/Mukhriddin19980901/Googlenet/blob/main/accuracy.png" width="600" height="600" />

