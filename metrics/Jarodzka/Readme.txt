About the Data
The groundtruth for all the three types of models are provided here: One can find 40 saliency maps to evaluate model two and 40xM observers scanpaths to evaluate model three, as well as 20 saliency maps to evaluate model one. Each of the stimuli is watched by 40-42 observers (information on observers are in excel file). The tools to read each of the three types of data(saliency maps and scanpath text files) can also found in the matlab file called "parseSalMapScanpaths.m".
*** The ground-truth data for model type 1, is organised into a binary file " SHE_<ImageNumber>.bin" containing float values (4 bytes), depicting the Heat Map in the equirectangular space. The Heat Map data is organised row-wise across the image. The minimum value of saliency is 0 and the sum of all image pixel saliencies equals to one.
*** The ground-truth data for model type 1, is organised into a binary file " SH_<ImageNumber>.bin" containing float values (4 bytes), depicting the Heat Map in the equirectangular space. The Heat Map data is organised row-wise across the image. The minimum value of saliency is 0 and the sum of all image pixel saliencies equals to one.
*** The ground-truth data for model type 3, is organised in a text file named "SP<ImageNumber>.txt". Each line contains a quadruple vector that indicates the Fixation Number, Fixation-Time,  X-Position (Equirectangular) and Y-Position (Equirectangular) respectively. The fixation number increments serially for a particular observer and resets to 1 when we reach the next observer, after all of the fixations of the given observer are completed. The fixation time is indicated in seconds and X and Y positions are indicated in pixels (of the respective Equi-Rectangular image). 

Visualisation of the data
Each type of data is also visualised by either a heatmap (for models 1 and 2) or using a scanpath number (for model 3). The visualisation of the heatmap is stored as a "red-blue" colormap ".jpg" where the hottest red regions indicate areas of extreme saliency and the blue areas indicate least salient regions. The white regions are those with intermediate saliency values. In case of a scanpath small numbers indicate the order in which a particular user fixated on that spot. For example "32" indicates that the user made his 32nd fixation in that area. Each observer is represented by a unique colors. So you may observe about 40 different colors each prtaining to a different observer and numbers which increment to indicate the order in which that particular spot was fixated.

Organisation of the data
There are four folders on the FTP called Scanpaths, Images, HeadSalMaps and HeadEyeSalMaps respectively. Images folder contains the original 60 images whose attention data is examined. All images are under the CC copyright and you are required to distribute the Images_Copyright excel file to provide appropriate credit to the photographers. The scanpaths folder has the text files containing the scanpath and the sample scanpath drawn on the equirectangular jpeg image. The HeadSalMaps and HeadEyeSalMaps contain the saliency map for the head only and the head+eye conditions respectively. There are 20 images where the "head only" attention data is computed and 40 where the "head+eye" attention data is computed.

Using the parsing function
This function "parseSalMapScanpaths.m" parses the scan-paths or the saliency maps. The function has a signature "function [result] = parseSalMapScanpaths(path,imgNum,typ)" and takes in three input arguments path,imgNum,typ and gives an output argument result.
path : This is the file path on your server or local machine that contains the data that was downloaded. This must be the root folder containg the 4 subfolders : Scanpaths, Images, HeadSalMaps and HeadEyeSalMaps
imgNum : Use a plain integer 0-10 here to indicate the image number which you want to examine. Please remember that certain images be used for "head-only" condition and others can be used for the other models.
typ: This can be either 0/1/2 based on whether you want to test the "head only","head+eye condition" or the "scan-paths" respectively.
result: in case the typ=0 or typ=1, then then the result contains an MxN equirectangular format saliency map where MxN is equal to the dimensions of the equirectangular image.
		in case the typ=3, then we obtin a scanpath as the resut. This is a :x5 vector where the first column indicates UserID, the cond column indicates the fixation number(counted for every observer serially), the third column holds the fixation start time (counted from the instant the stimuli was shown), the third and fourth column indcates the fixation point in X,Y coordinates of the equirectangular image.	
A sample call can be [img] = parseSalMapScanpaths('F:\VR\GazeData',2,1);

DISCLAIMER
Kindly note that this 60 image dataset is only a "Training-dataset" that participants may use for training their algorithms and self-evaluation of performance. Any results reported on this data-set is not considered as an official benchmark score. For official scores, we will test your algorithms (binary submissions) on our 35 image "Validation-dataset" that has not been released so far.

Please refer to the FAQ document for further details or feel free to write to salient360@univ-nantes.fr


