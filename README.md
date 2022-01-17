#  Task 1.1 - Data Visualization
Here you have to visualize the results on the images. In the folder "Data Visualization" there are two folder
1. Images - Contains images named as 1.jpg, 2.jpg and so on
2. Data - Contains metadata in form of JSON file named as 1.json, 2.json and so on

The naming is consistant between them for example 1.json is the metadata for image 1.jpg. In the json file you will find normalized xy  co-ordinates of polygons and class of the polygon., alongwith which side of car it is (passenger side, front side, driver side etc.) 

For normalizing the co-ordinates following formula is used
normalized_x = ( x / image_width ) * 100
normalized_y = ( y / image_height ) * 100

Hint --> masks and alpha blending
### Following is the breakdown of the task:
1. You have to go through json and figure out which fields to use and which field is for what.
2. You will need to denormalized the polygon co-ordinates to use them to plot on image
3. This polygons need to be plotted on image with transparancy and border highlighted with darker shade of same color (see an example of how it should look below)
4. You have to do total 2 types of visualiztion
	1.	Where only transparnt polygons are drawn
	2.	where bounding boxes are drawn around them and class of polygon   displayed some on corner (can choose any corner)

Examples
Visualiztion of first type (Same with not bounding box)
Notice that the border of area and bbox have similar color but darker shade
![Visualiztion of first type](https://drive.google.com/uc?export=view&amp;id=1fb8BNtQa2Sde2LwcjuVLb8_oarcR17Jg)
Visualiztion of Second type but without the confidende score
![Visualiztion of Second type](https://drive.google.com/uc?export=view&amp;id=14YRKrlBWK--mm_5ct7abpPG3yGjKp3og)

Note - How  much transparnt the polygon should be as per your taste but the parameter should be tweakble. also implementation using using low level function or from scratch will be prefered over high level one line implementation (e.g. detectron2 visualizer class)

#  Task 2.1 - Damage Classifer
**Objectives :-** You have to train a binary classifier to check if the car is damged or not DataSets you can use 
 https://www.kaggle.com/anujms/car-damage-detection
You can label more data or use any other dataset you may find on internet it's upto you
You can use any framework you want pyTorch and TensorFlow
1. create a notebook on kaggle or colab (both fine)  it should have data preperation, training and evaluation code in it
2. Provide the Class weights drive link will work (bouns point if you could directly download the class weights without requiring to manually set)
3. Put the approach you have taken, augumentation policy you used, evaluation methods you have used and why etc in the notebook as markdown