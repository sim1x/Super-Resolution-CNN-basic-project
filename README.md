# Super-Resolution-CNN-basic-project
This repo contains a simple and basic project that uses a neural network to upscale the resolution of an input image.
To test the code you should:
1) run "Scuola_HighResNet.py" which will download the dataset and train the neural net
2) run "HiResNet_inference.py" to see the net used on the test images

The model is full convolutional and it takes 1'/1'15 minutes to train. For this project I decided to go with the dataset STL10, available from torchvision.dataset. I'm using 100 images as total dataset (with resolution of 128x128) and then i split it in 60 training images, 20 validation images and other 20 test images. Inside the folder you will find the script "Scuola_HighResNet.py" that trains the network and uses it on some of the validation images. Then you will also find "HiResNet_inference.py" in which i am loading the saved model and test images from previous script and using the trained neural net on these images.
The following image is an example of what you can expect by this network:
<img width="1172" height="777" alt="esempio" src="https://github.com/user-attachments/assets/cf9a957f-2b35-43ab-b686-1f3a7f254dbd" />
As you can see it is not perfect but i think it's quite interesting considering the small dataset used (100 images) and the actual neural net: it consists only of a bunch of layers.
