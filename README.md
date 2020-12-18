# Anamoly-detection-in-construction-site

Dataset:
Total number of images = 810.
Training data =675 (80%).
Validation data = 135(20%).
Number of classes =5 (Worker,  Strap, Hardhat, Hook, Harness).

WorkerDetection is trained on dataset annotated with VGG Image annotator

WorkerDetection1 is trained on dataset annotated with Labelme annotator

Integration model is the integration of Mask RCNN and Linknet

Integration file includes the final results (classifying worker as safe or unsafe based on adjacent matrix)

