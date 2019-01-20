# Saliency in time

The goal of this project is to try to simulate plasuible scanpaths, fixations and saccade, in free viewing condition. 


<img src='https://raw.githubusercontent.com/rAm1n/salincy-in-time/master/imgs/1065.jpg'></img>



## Models

Two model has been implemented based on foveatin technique. First iteration is always with full reslution image and it will follow with foveated inputs based on previous fixation chosen by the model. 



#### A)

**A** is Based on learning quantizing saliency volumes with q=400ms.


<img src='https://raw.githubusercontent.com/rAm1n/salincy-in-time/master/imgs/model-a.png'></img>



#### B)

**B** is an Encoder-Decoder model based on use of normal and bidirectional ConvLSTMs as decoder. 

<img src='https://raw.githubusercontent.com/rAm1n/salincy-in-time/master/imgs/model-b.png'></img>

### Evaluation

 Please make sure to checkout [SEQUAL](http://github.com/rAm1n/SEQUAL) to find datasetes and metrics used in this project.
