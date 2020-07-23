We have four folders where our all images that we have used for training and prediction are saved.
1.	training_images folder contains all the ten synthetic (RGB) images that we have generated for our work.
2.	disparity_images folder contains all the ground truth Images corresponding to all the synthetic (RGB) images.
3.	testing_image folder is the folder that can be used to save an image that we want to use for prediction
4.	test_images folder where we have saved our new  Synthetic (RGB) Images that we have used for more generalization purpose.
5.	test_disparity _images folder contains ground truth image corresponding to the new image for generalization.


Note: Out of 10 Images that we have created, we can take any number of images for generating dataset to train our neural network and also we can take any different Image for prediction, we can save the synthetic image that we have to use for prediction in the testing_image folder.

We can follow the same process if we want to use some other image for prediction.

Test_image and Test_disparity folder consists images which are specifically created to check the robustness (for generalization purpose) of our network by checking its prediction values ,that is why we have not used this image for training and it is advised to use one of this image to check how well our network generalizes.

We have already created five new test images and its corresponding disparity images; in case we want to use one of the images to generate dataset for training and we want to use it for training then we need ground truth data so it is already available but if we don’t want to use these images for training then we can use one of these synthetic (RGB) Image and place it in testing_image folder and can do predictions on that Image. 

We can save all the folder containing images on our desktop using same name or different, just be careful to change the path while generating dataset for training and prediction. This step is very important in preparing dataset.

We have to two scripts inside master branch:
render_script:, we used this script in blender to generate synthetic (RGB) images.
Disparity:, we have used this script to generate ground truth images corresponding to the synthetic rgb images from positional images.

But we don’t need run these scripts as we already have the images but in case we need to generate new images we can use these generate for that.

We can get all the folders on my desktop: 
                                                 *********************************

When we are performing the experiments it is advised to store all the folders on desktop and all the other folders that we are going to discuss now in one main folder with some name like “all_files” as these folders contains all the files that we have implemented and the above folders are just of Images.
Let’s start discussing these folders one by one everything is stored in master branch only:



1.Preparing_dataset
2.Models
3.Results

Note: Store the main folder in some directory.

First we have to run the files of preparing_dataset sub folder in order to generate the dataset for training and prediction.
To prepare dataset for training and prediction do the following steps:
1.	First go to all the .py files in the preparing_dataset sub folder and change the path: 
config.readfp(open('/home/rgupta/all_files/preparing_dataset/path.properties')) into
config.readfp(open(‘your path’))  “please remember we want to read “path.properties “files so path must end with path.properties. 
path.properties is the file where all paths will be written to save the datasets and read the images that we want to use to generate the dataset.

Note: remember all the files (python files and path.properties) names are in lower case.

2.	Go to the file path.properties and mentioned the below path for respective section (section is enclosed in squared bracket In our path.properties file there is [PREDICTION], [TRAINING] and many other sections, basically these are called section and we can use them to specific path that we want to call in some other file, names of section are always written in capital letters and we defined them for each lens type.

In the file path.properties there are two sections with name [PREDICTION] AND [TRAINING] .
In TRAINING section we have to specify the path of the folder where images for training are stored and path for its corresponding ground truth Images.
In PREDICTION section we have to specify the path of the image on which we have to do predictions and also the path for coordinate value list which we can use to generate predicted image from estimated disparities.

We can save the coordinate value list in the folder where we want to save the results of our model and it depends where we want to save it, but we have to make sure to remember it as it is very important for generating image back again from disparity values

When we are generating training data in the path.properties file we can see that we can store the path of all the different input data type that we have used in our work.

It is advised that for every input lens that we are choosing for our work we can make one main folder named “dataset” and in that main folder we can make two sub-folders with name train and prediction and inside train sub folder we can again make sub-folders depending on the type of lens we are using for e.g. We are two-lens then inside train sub-folder we can make left, right and disparity folder(with names left, right and dis_images1) and inside prediction sub-folder we can make sub-folder left and right(with name left and right) and we can then give their path in the path.properties file and for all other lenses we can follow the same process because later we want to take all these files to train and predict our network and we have designed our network in such a way that it reads one main folder and then the sub-folders .

Our all codes are well-documented for better understanding. Below we have explained path.properties files procedure in more detail.

We can give the name of the folders for all type of lenses that we are using as mentioned above for two-lens and please see path.properties files to give the names as we have given so that we don’t have to change in the model files while running them if the names are different then while loading the data from the folder we have to specify the other names that we have used.

Please remember Dataset folder will contains dataset for a particular lens type, if we want to run another lens type then we have to make another folder with some other name and we can then specify that path in our model files to run the model with that lens_type. 


For e.g.: In [TRAINING TWO] section there is a left path, right path and disparity path as we are cutting two adjacent lenses and the corresponding disparity lens in that. Similarly for four lenses we have middle, left, right, top, bottom path as we have four immediate neighbours so we can specify path where we want to save these lenses and for that we have to make all these different left, right, top, bottom and disparity folder and so on and we will follow the same process for every input lens that we have used whether it is six, twelve seven, three. Seven and three are the lenses that we have stacked we have already discussed about this in our thesis.

When we are generating data for prediction we can see in our path.properties file there is:
[PREDICTION_TWO] section here also we left, right path and we have coor_values path so in left and right path we will store the path of the folder where we want to store both the immediate left and right lenses.
Here we have extra path for coor_values. Coor_values is the path where we have stored all the coordinates of the main lens so that we can use that coordinates to generate the prediction Image as discussed above.


Note: main lens is nothing but a single lens that we have selected and all the other lenses are neighbours that we have used for matching (in two lens main lens is left and right lens is its neighbour and corresponding to left lens we have its ground truth lens also, similarly in four lens input data middle lens is the main lens and all the left, right ,top and bottom are neighbouring lenses that  we have used for matching  and corresponding to middle lens we have its ground truth lens  for six, twelve and other lenses  the working is same  ) 


3.	How to execute the script:
For running the scripts from console enter following command:
python script_name argument (script name is the name of files that we want to run for example if we want to generate training data then put “cutting_images_for_training.py”, all in lower case, similarly we can put other names and run the program) and argument is if we want to generate data for two lens or four lens and so on. So in argument put:
•	two  (for two lens)
•	four  (for four lens)
•	six  ( for six lens)
•	twelve ( for twelve lens)
•	seven (for horizontal seven lens data)
•	three (for horizontal three lens data)
Note: for color_augmentation file we have just used two lens input data and for relevant _lens_training file we have used two and four lens input data. So if we are running these files then they will only work with lenses that we have used for them.

“This is all about the preparing_dataset”

                                                                     ***************************


Now we will look at our second sub-folder which is models, here we have saved all our models that we have used to estimate the disparities. We have 8 files in this sub-folder each files belongs to the model that we have used let’s look at them:
1.	First file is model1_transpose. In the file we can find this:
""" path where data for training and prediction are saved """
data_set_path = "your path"
Here we have to specify the path where our main folder dataset where all the training and prediction data is saved. There are several path in the file that we have to save in some directory and to save all the path please use same folder so that it would be convenient to track and use them later. We can save all the paths where we want to store all the results.
We have made this model (custom model) for two, six and twelve lens.
How to execute the script:
For running the scripts from console enter following command
python file_name argument (two,six, twelve)

2.	If we are using second file which is same model as before but with slight variation
We have used second file for just four-lens Input data so here also we have specify the path like before to fetch the dataset so that we can feed the network the data for training and prediction. There are certain paths which we need to save in some directory for generating the prediction or output of the network.Code is documented properly so we will get more Idea from there.
        How to execute the script:
For running the scripts from console enter following command:
 python file_name argument (four)


3.	 In third file we have implementation of different network (U-net) which can be used for six and twelve lens input data and the ones we have discussed above we have to specify the path of the main folder where our training and prediction dataset is saved. 
How to execute the script:
For running the scripts from console enter following command:
python file_name argument(six,twelve)

4.	 In Fourth file we have our another network (resnet network) and for this network we can use two, four,   six, twelve lenses and here also we have to specify the path where dataset are saved for training and prediction like we have done before.
For running the script from console type:
python file_name argument(two,four,six,twelve)

5.	 Fifth file contains implementation  of our resnet  network specifically used for horizontal seven and three lens. Here also we have to specify the path where our dataset is saved and for details see Implementation.
For running the script from console type:
python file_name argument(seven, three)


6.	Our six file contains the implementation of our state -of-art network which we called epinet that we used for comparison and we can use it for two, six and twelve lens although we have just used two-lens result for the comparison in our work. Rest of the things are same we have to specify the path where we want to save the graphs, log, checkpoint and other things that we have used in the implementation.
For running the script from console type:
python file_name argument (two, six,twelve)

7.	Our seven file also contains the implementation of epinet network specifically for seven and three lens input data.
For running the script from console type:
python file_name argument (seven,three)


8.	Our eight file is implementation of opencv  stereo block-matching algorithm and for this file we can use two-lens prediction lenses that is left and right as we are not doing any training so we don’t require any training data and we can specify the path accordingly.
For running the script from console type:
python file_name 

                                                        ******************************

Our third sub-folder which is results folder can be used to generate the results and Images that we have seen in our thesis report all the files in this sub-folder is well and clearly documented.
But here I would briefly explain the significance of each file for better understanding.

1.	First we have to run the file named prediction_image  and in the file we have to load certain path in order to generate the Image, which we called predicted Image.
For.eg. :  If we are running some model with two-lens inputs data then we have to save the estimated disparities and using that disparities and the coordinates of our RGB image( which we have used for  prediction ) that we saved while creating dataset for prediction in “ preparing_dataset” sub-folder we have a file “cutting_images_for_prediction” we have saved the coordinate values. we have to load that coor_values from where we have saved them into this file and then We can generate prediction image to visualize our disparity values and to compare our predicted Image with the original ground truth Image.

As we already know that we have cut down rectangular boxes and inside of our rectangular boxes we have our hexagonal lens structures so to see those structure in our output Predicted Image we have to run another file with named “hexagonal_patch” which we have created a small 35 by 40 box/patch  containing hexagonal structure and then we can save that image somewhere on the desktop or in some directory and while running our prediction_image file we also have load the hexagonal small Image to see the hexagonal structures in our predicted Image

Both the files are well documented for clear understanding.
 How to execute the script:
•	First run hexagonal_patch file from console using command
Python (file_name) hexagonal_patch.py
•	Then run prediction_image file using command
Python (file_name) prediction_image.py

Note: Every time we need to generate predicted image after estimating disparities for different input data type lenses we will run prediction image file and before running prediction_image file, we have run hexagonal_patch file.But if we need to generate predicted image for three, seven lenses we have to run prediction_image_horizontal_seven file for seven lens type and prediction_image_horizontal_three for three lens type and before running these files also we have to run hexagonal_patch file. Execution proc3.ess is same we just to change the name of the file and for generating predicted image from opencv stereo block matching algorithm we have to run prediction_image_stereo_block_matching file.

   
2.	The second result in the file that we need to generate is difference image and for that we have to run the file difference_image.py . In this file we need to check the difference between our original ground truth image and the predicted image. Here we need to remember that our original ground truth image has size 3500 by 3500 pixel but for prediction we are not taking all the pixels so while comparing our predicted image with ground truth image we simply use those pixels we have used for generating prediction image and to do so we have to run two files:
•	rectangular_cutouts 
•	ground_image_hexagonal
First we have to run rectangular_cutouts files to take the only those rectangular boxes that we are taking for prediction and here also we have to save the coordinates of the rectangular boxes as we have to again generate the full ground truth image with the hexagonal lenses.  See code for detailed documentation.
For execution run:
python rectangular_cutouts.py

Then we need to run ground_image_hexagonal file because we need to see hexagonal structure and here we will load all the rectangular boxes/lenses/patches that we have cut in rectangular_cutout  file and also we need hexagonal small image that we have already stored in some directory we will load it from there.
As stated above our implementation is well-documented so we can read it for more understanding.
For execution run:
python ground_image_hexagonal.py 

3.	Next file that we have to run is disparity_error_com to generate our disparity vs. error and pixel vs. error graph using ground truth image that we have created by running ground_image_hexagonal.py file and predicted image.

In the file we have written everything that needs to be done to create the graph as for models and for the methods that we have used for comparison we have to run the file and there are slight changes that we need to make depending on for which method we need to execute it .
For execution run:
python disparity_error_com.py 

4.	The next file that we have to run is metrics_values and metrics_values_block, from both of these file we will run file at a time so if we areusing our models and open stereo block matching method than we will run metrics_value file to generate mean squared error and bad pixel ratio and if we are using our another Block-matching method (reinhard koch) then we will use metrics_values_block file for generating mean squared error and bad pixel ratio.
We need to change the path in the files before executing it.
For execution run:
python metrics_values.py /python metrics_values_block.py

Note: If we want to compare  mean squared error(mse) and bad pixel ratio(bpr) values of different input type lens or different method ,we are saving the mse and bpr values in a text file in appending mode so we can use the same file and we can give the names of the input lenses or the method that we are using in the text file and we can save it in some directory and later we can use the text file for plotting the mse and bpr values in a graph. Our next file which will be our last file is doing this only it is plotting the mse and bpr values in a graph which we have already seen in our thesis.

5.	Our last file which we want to run for plotting mse and bpr values in a graph is plotting_mse_bpr file
As discussed in this file we are plotting the values of mse and bpr and in the file if we want to plot mse graph if we to specify the path of  mean squared error text file and if we want to plot bpr values then we have to specify bpr text file path and accordingly we have to change the title or the x-label  and the name of the graph that we are saving.
  For execution run:
  python plotting_mse_bpr.py

6.	There is one more file with name pixel_com,  which we can run to compare the number of pixels used in our method and Block-matching (raw and cross-check) method.4

  For execution run:
  python pixel_com.py







To import all the libraries we need to install in console:

1. pip install numpy.
2. pip install opencv-python.
3. pip install natsort.
4. pip install albumentations.
5. pip install matplotlib
6. pip install configparser.
7. pip install os-sys.
   8. pip install kears
   9. pip install tensorflow
10. pip install theano
