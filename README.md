#  Damage Detection and Visualization
1. Car damage visulaization using given metadata for sample images
2. Damage detection, bounding box and labeling using Mask R-CNN 

#  Damage Classifer
Base Model - DenseNet121 (pre-trained on ImageNet data)

Following additional layers are introduced to increase prediction accuracy - 
	1. Data Augmentation (Flip, Rotation, Zoom)
	2. Rescaling
	3. Dense NN
	4. Top Dense NN with softmax activation