# Complex Binary Classification With Deep Learning

## After learning the softmax regression algorithm, I was confident that I could design any classification models, no matter how many labels and features are present. However, after attempting to figure out a way to classify the following dataset, I was humbled.
<img width="545" alt="Screen Shot 2022-06-28 at 8 03 59 PM" src="https://user-images.githubusercontent.com/102645083/176343140-fda8190b-d58a-4eb8-8b93-be2edaa1f610.png">

## While we could easily recognize the windmill-like pattern right away, it's impossible to design a model that could figure it out using just classic ML algorithms. I initially thought that using KNN may do the trick, but the predictions near the outliers and the center would become a mess. 

## This led me to the topic of Artificial Neural Networks, which is a series of deep learning algorithms that simulates how the human brain process information, and it's especially useful for modeling non-linear statistical data, like this project. Every network starts with the input layer, where the initial data enters the workflow as Artificial Input Neurons. They are then inputted directly into the first of possibly many fully-connected hidden layers, where neurons go through a nonlinear transformation by taking in a set of weights and bias, and are outputted right away or through an activation function. This process applies to every hidden layer, and the outputs from the final hidden layer is called the Output Layer, where the conclusion can be made, and it can vary from percentages for regression problems to labels for classification problems. This entire workflow is called Forward-Propagation, which is the process of calculation and storage of intermediate variables for a neural network, and every layer's input is the previous layer's output. The weights and bias of each layer are updated through every iteration, and it's important that we include a loss function after Forward-Progagation to compare the predictions to the actual values. 
<img width="626" alt="3-intro-deep-neural-networks" src="https://user-images.githubusercontent.com/102645083/176347293-f65a8a75-75bd-4521-9371-04941e70a4ee.png">

## The next part of this algorithm is called Back-Propagation, and I struggled a lot with understanding it. In classic ML algorithms, we never have to deal with an enormous amount of variables (weights and bias that we are trying to find), and performing partial derivatives and optimization using gradient descent were quite straightforward. However, we can't do that for neural networks. If we think of an entire network as a function, every neuron / variable directly affects the final outcome. Complex network models can have millions of them, which means that by just performing partial derivative on just one neuron, we need to take into account for the chain rule caused by every single other neuron. Doing such is virtually impossible, which is why Back-Propagation is so important. By using what we've already calculated using Forward-Propagation, we can find the error rate (dC/dz) for each neuron of every layer, which makes it very simple to find the partial derivatives in term of the weights and bias. In simpler words, the two propagations work simultaneously to fine-tune the weights and bias of the entire network through each iteration.
<img width="879" alt="Screen Shot 2022-06-28 at 9 27 44 PM" src="https://user-images.githubusercontent.com/102645083/176351694-baac47c0-4d72-4dd8-b895-21257f77aec4.png">

## By understanding how this algorithm works, I was able to work with a model that can classify this complex dataset. I first used only NumPy to come up with a "hard-coded" version of this model. To do so, I first initialized a network using my desired amount of parameters. It's important that we set the initial weights using random values from a fixed range because we don't want to start with the same value through every iteration. After, we create the Forward-Propagation function, which uses tanh as the first hidden layer's activation function, and sigmoid for the output layer after. We then create a loss function, which uses cross entropy because all of our labels are 0's and 1's. Lastly, we create the Back-Propogation function, which uses the values found in Forward-Progation and find-tune the two set of weights and bias. By performing these three steps 100,000 times, we can see that this model becomes increasingly more accurate after more trials and errors.
<img width="316" alt="Screen Shot 2022-06-28 at 10 01 29 PM" src="https://user-images.githubusercontent.com/102645083/176355414-a2387ca3-a0ff-4f25-8a3b-e2b8f3012335.png">

## Since all the predicted labels are decimals between 0 to 1, it's important that we round them to which ever integer they are closer to. After doing so and plotting out the results, it's safe to say that this model is quite accurate and ignored a majority of the outliers.
<img width="540" alt="Screen Shot 2022-06-28 at 10 03 05 PM" src="https://user-images.githubusercontent.com/102645083/176356094-48dcaa34-b92c-46f5-ae98-c802bf6b7409.png">

## Building this model using TensorFlow is much easier. We can initalize the model using "model = tf.keras.Sequential()" and setup the layers and activation functions using "model.add()". We then config the loss function / optimizer / metrics using "model.compile()" and start the learning with "model.fit()". After spending some time on hyperparameter tuning, I found that a validation split of 0.2, epochs of 200, and batch size of 10 resulted in the most desired outcome after testing the model using "model.predict()". The result was the same as the previous method.
<img width="540" alt="Screen Shot 2022-06-28 at 10 03 05 PM" src="https://user-images.githubusercontent.com/102645083/176357304-489a10e2-0a5e-4778-aafd-2c800e8f8321.png">

## By using "model.get_weights()", we can see the weights that we've been looking for this whole time.
<img width="943" alt="Screen Shot 2022-06-28 at 10 21 05 PM" src="https://user-images.githubusercontent.com/102645083/176357748-af64dc4b-807d-4924-aa82-ecc43f312a1e.png">

## At the end of the day, this classifiation problem still requires the use of a classic ML algorithm, which is logistic regression. As complicated as neural networks sound, it's just a way for a machine to analyze the original data from primitive to more advanced features before being concluded with a regression or classification algorithm.
