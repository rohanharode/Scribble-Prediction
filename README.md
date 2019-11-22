# Scribble-Prediction-CNN-GAN

This project aims at predicting scribbles from canvas and also using GAN generated images.

Idea was inspired from Quick Draw dataset.


Overview - 
1. Trained a GAN model on few categories of objects from Quick Draw dataset. Save generated images in directory.(scribbleGAN.py)
2. Built a training model using a ConvNet Model (train.py).
3. Developed prediction model (server.py) which takes input from either canvas or GAN category dropdown. 
4. UI was created using HTML, JavaScript, JQuery and Flask App. 


![GAN-Sample](https://github.com/darklord0794/Scribble-Prediction-CNN-GAN-/blob/master/gan-final_chart.png)

![GAN-Loss](https://github.com/darklord0794/Scribble-Prediction-CNN-GAN-/blob/master/gan-loss.png)

![WebApp](https://github.com/darklord0794/Scribble-Prediction-CNN-GAN-/blob/master/webapp.png)
