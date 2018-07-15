# 2018DSB
2018 Data Science Bowl 2nd Place Solution

My solution is a modification of Unet. To make Unet instance‐aware, I add eight more outputs 
describing the relative positions of each pixel within every instance as shown in the images below.  In my final model version, the entire network structure of Unet before the output layers was replaced by the 
pre‐trained Mask‐RCNN feature extractor (P2 as in Matterport version of MaskRCNN) for better 
performance. 

https://github.com/jacobkie/2018DSB/blob/master/imgs/0.png

The relative position masks are shown as below:
https://github.com/jacobkie/2018DSB/blob/master/imgs/1.png
https://github.com/jacobkie/2018DSB/blob/master/imgs/1.png
 
Codes in utils.py, parallel_model.py, params.py, visualize.py, model_rcnn_weight.py 
are partly adapted from Matterport Mask_RCNN 
(https://github.com/matterport/Mask_RCNN) which is under MIT license. I also used its pre‐
trained weight on MS COCO (https://github.com/matterport/Mask_RCNN/releases).  

Four sources of data were used:  
1. The Revised Train set(https://github.com/lopuhin/kaggle‐dsbowl‐2018‐dataset‐fixes) 
2. 2009 ISBI (http://murphylab.web.cmu.edu/data/2009_ISBI_Nuclei.html) 
3. Weebly (https://nucleisegmentationbenchmark.weebly.com/) 
4. TNBC (https://zenodo.org/record/1175282#.Ws2n_vkdhfA)  
 
Some masks of the 2009ISBI data set are manually modified. 

To train from scratch
1. correct directory addresses of stage1 train set and stage2 test set accordingly in params.py
2. run eda.py  to load images&masks and save into pandas dataframe from stage1 train set and stage2 test set.
3. correct directory address and run 2009isbi.py weebly.py TNBC.py
4. run resize.py to create image pads of 256*256 size for all the datasets above.
6. run train_ext.py to train from pretrained weight on MSCOCO
7. correct directory address of weight_dir (where the model weight is saved) and run predict_auto.py to predict stage2 test set at four zooms (1/4, 1/2, 1, 2), and generate instance masks accordingly. 
8. correct directory address of weight_dir and run submission.py to combine instance masks from four zooms and mask submission file. 

Or you can use my pretrained weight in the cache folder to make predictions directly. 
