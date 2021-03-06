# Motivation
This project wants to use long text description as label to extract high level features from images with DNN. The high level features are intended to represent interest of fashion for posters which is latent feature we want to estimate. 
The research is temporrally paused and I uploaded codes here.

# Extracting clothes coordinates features by Image and Text Co-learning strategy
This python code learns fashion coordinates (from wear.jp) and give two level topics (i.e. image and text) recommendation.
CNN trains and predicts categories and colors in each coordinate. Then Grade of Membership model extracts the features (memberships) of different choices on colors and categories.
LDA deals with the text descriptions of coordinates and try to find meaningful topics of these coordinates.
Finally, this program can give two levels features from the fashion coordinates (can be applied to any other image-text posts dataset)

The difficulty is to find the most intepretable number of memberships and topics. Also, tuning CNN is another hard task.
But with high performance of CNN, the model can be widely applied to such Instagram-like dataset, which will be useful to understand community interests.

## 1. WearjpScraper
This python script retrieves images and text descriptions from wear.jp
Including:
- Coordinate Image
- Coordinate Tags (e.g. Top - Brand & Color, Bottom - Color, Accessories - Color, etc)
- Post Title
- Post Descriptions
- Post URL
- Post No. Favs
- Post No. Likes

The script can scrap by filters. Just replace the coordinate urls in the script.


## 2. CNN
I used fashionnet pretrained model to train CNN. It has two branches for category and color recognizing. They are: Category/Color of Tops and Bottoms. The categories and colors are scrapped from Wear.jp.

The training accuracy acheived 50% or so out of 10+ categories with branch-training strategy. I wonder splitting the brach from middle layer may achieve better accuracy because the model would share some low-level features at the beginning. Intuitively speaking, the model first learn the features for both tasks (category and color), and after learning the location of object, the model can focus on different tasks on the correct area.

## 3. LDA/GoM
The two models are hierarchical bayesian models for extracting features of image and text. They are OK to run, but slowly. I plan to implement PyRo to increase the speed, which uses PyTorch as backend and thus can be computed by GPU.

Basically, I chose TensorFlow as my model's main framework. But in the future, I will go to PyTorch since it is easier to implement and is used as Bayesian inference backend as well.

GoM, Grade of Membership model, is like a hierarchical topic distribution. In this model, each membershio includes a series of different topic distribution for each choice of item category and color. Therefore, GoM can learn concurrence of item selection (i.e. clothe coordinates) and thus it reserves coordinate information.

LDA is used as basic Topic Model to extract coordinate description features.

# Model Plate Diagram


![The model aims to integrate text and image features together based on Polilingual Topic Model. But I tried to implement image and text feature extraction separately.][graph]

[graph]: https://github.com/yanzhou01/twoLevelLearning/blob/master/0_etc/preview.svg