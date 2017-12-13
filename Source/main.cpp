//IO
#include <fstream>
#include <stdio.h>
#include <iostream>
//OpenCV
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"

#include "Utilities.h"
#include "Page.hpp"

using namespace std;
using namespace cv;

#define NUM_IMAGES 25
#define NUM_TEMPLATES 13
//Obtained through trial and error
#define PAGE_THRESHOLD 160
#define CIRCLE_THRESHOLD 10

void loadImages();
void getTemplateInfo();
void performCalculation();
void performanceTest();
Mat matchMultiple(Mat src, Point2f corners[], int index);

//stores the sample and template images
Mat* images = new Mat[NUM_IMAGES];
Mat* templates = new Mat[NUM_TEMPLATES];
//sample image of blue circles
Mat blue_pixels;
Mat home;
//stores the corner points of each template
Page* template_pages = new Page[NUM_TEMPLATES];
//stores the horizontally joined output images
Mat* output_images = new Mat[NUM_IMAGES];

//performance variables
int ground_truth[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,2,3,5,4,7,9,8,7,11,13,12,2};
int results[25];

int main(int argc, const char** argv)
{	
	cout << "Recommend running in Release" << endl;
	
	performCalculation();
	cout << "\npress space to view image output" << endl;
	int choice;
	int index = 0;
	do {
        imshow("Welcome", home);
		choice = cvWaitKey();
		cvDestroyAllWindows();
		switch (choice)
		{
		case 32: //space
			if(index < 25) {
				imshow("Image " + to_string(index),output_images[index]);
				index++;
			}
			break;
		default:
			break;
		}
	} while (choice != 27); //close on esc
	
	home.release();
}

void loadImages()
{
	//IMAGES
	//Store image file paths
	for (int i = 0; i < NUM_IMAGES; i++) {
		String path = "";
		int imageNum = i + 1;

		if (i < 9)
			path = "../Media/BookView0" + to_string(imageNum) + ".jpg";
		else
			path = "../Media/BookView" + to_string(imageNum) + ".jpg";

		images[i] = imread(path, CV_LOAD_IMAGE_COLOR);
	}

	//TEMPLATES
	//Store template file paths
	for (int i = 0; i < NUM_TEMPLATES; i++) {
		String path = "";
		int imageNum = i + 1;

		if (i < 9)
			path = "../Media/Page0" + to_string(imageNum) + ".jpg";
		else
			path = "../Media/Page" + to_string(imageNum) + ".jpg";

		templates[i] = imread(path, CV_LOAD_IMAGE_COLOR);
	}

	//Load blue pixel sample
	blue_pixels = imread("../Media/BlueBookPixels.png", CV_LOAD_IMAGE_COLOR);

	//Load home image
	home = imread("../Media/home.png", CV_LOAD_IMAGE_COLOR);
}

/*
 * Loops through the template images, identifying their corners and storing this data
 * in a Page object.
 */
void getTemplateInfo()
{
	for (int i = 0; i < NUM_TEMPLATES; i++) {
		Mat cur_template = templates[i];
		Mat projected_image = backProjection(cur_template, blue_pixels);
		Mat binary_template = manualThreshold(projected_image, 20);

		Point topLeft, topRight, bottomLeft, bottomRight;
		detectTemplateCorners(binary_template, topLeft, topRight, bottomLeft, bottomRight);
		// Store the 4 points where the mapping is to be done , from top-left in clockwise order
		template_pages[i].corners[0] = topLeft;
		template_pages[i].corners[1] = bottomLeft;
		template_pages[i].corners[2] = bottomRight;
		template_pages[i].corners[3] = topRight;
		template_pages[i].width = cur_template.cols;
		template_pages[i].height = cur_template.rows;
	}
}

/*
 * Performs all the calculations on the images to identify their
 * page.
 */
void performCalculation()
{
	loadImages();
	getTemplateInfo();

	for(int i = 0; i < NUM_IMAGES; i++)
	{
		cout << "[LOADING image" + to_string(i) + "] ";
		//Convert to grayscale
		Mat gray;
		cvtColor(images[i], gray, CV_BGR2GRAY);
		//Binary Thresholding
		Mat binary = manualThreshold(gray, PAGE_THRESHOLD);
		//Closings to remove page contents
		int iterations = 3;
		Mat closed = morphology(binary, MORPH_CLOSE, iterations);
		//Use mask to identify page region in bgr image
		Mat masked_image;
		images[i].copyTo(masked_image, closed);
		//Back project blue pixels
		Mat projection = backProjection(masked_image, blue_pixels);
		//Binary Thresholding
		Mat binary2 = manualThreshold(projection, CIRCLE_THRESHOLD);
		//Detect Corners
		Point topLeft, topRight, bottomLeft, bottomRight;
		detectCorners(binary2, topLeft, topRight, bottomLeft, bottomRight);
		//Store the 4 corners found
		Point2f inputQuad[4];
		inputQuad[1] = topLeft;
		inputQuad[0] = bottomLeft;
		inputQuad[3] = bottomRight;
		inputQuad[2] = topRight;
		//Draw circles on detected corners
		Mat corners = addCircles(images[i], topLeft, topRight, bottomLeft, bottomRight);
		//Perform template matching 
		Mat matched_template = matchMultiple(images[i], inputQuad, i);
		
		//Output
		Mat resized;
		resize(corners, resized, Size(), 0.5, 0.5, INTER_LINEAR);
		Mat output = JoinImagesHorizontally(resized, "Corners Identified", matched_template, "Matched Template");
		output_images[i] = output;
		cout << endl;
	}	
	performanceTest();
	blue_pixels.release();
}

/*
 * Function taken from chapter 8.6.3 of 'A Practical 
 * Introduction to Computer Vision with OpenCV' by
 * Kenneth Dawson-Howe.
 */
void performanceTest()
{
	int FP = 0;
	int FN = 0; 
	int TP = 0;
	int TN = 0;

	for (int i=0; i < NUM_IMAGES; i++){
		int result = results[i];
		int gt = ground_truth[i];

		//a page is visible and recognised correctly
		if(gt == result)
			TP++;
		//an incorrectly recognised page, where a different page was visible
		else FP++;
	}

	double precision = ((double) TP) / ((double) (TP+FP));
	double recall = ((double) TP) / ((double) (TP+FN));
	double accuracy = ((double) (TP+TN)) / ((double) (TP+FP+TN+FN));
	//double specificity = ((double) TN) / ((double) (FP+TN));
	double f1 = 2.0*precision*recall / (precision + recall);

	cout << "\nFalse Positives: " << FP << endl;
	cout << "False Negatives: " << FN  << endl;
	cout << "True Positives: " << TP << endl;
	cout << "True Negatives: " << TN << endl;
	cout << "precision: " << precision << endl;
	cout << "recall: " << recall << endl;
	cout << "accuracy: " << accuracy << endl;
	//cout << "specificity: " << specificity << endl;
	cout << "f1: " << f1 << endl;
}

/*
 * Matches all templates against a given image. 
 * @param src the given image
 * @param corners the 4 corners of src
 * @param index the index of src in the images array
 * @return matched_template the template which matched the given image
 */
Mat matchMultiple(Mat src, Point2f corners[], int index)
{
	double match_probability = 0.0;
	double match_result = 0.0;
	int template_index = 0;
	Mat transformed_image, matched_template;
	cout << "[";

	for(int j = 0; j < NUM_TEMPLATES; j++){
		cout << "==";
		//perform perspective transformation
		transformed_image = perspectiveTransformation(src, templates[j], corners, template_pages[j].corners);
		//convert image and template to edge images
		Mat image_edge = cannyEdgeDetection(transformed_image);
		Mat template_edge = cannyEdgeDetection(templates[j]);
		//perform template matching on edge images
		match_result = templateMatching(image_edge, template_edge);
		//identify which template has the highest match probability	
		if(match_result > match_probability){
			match_probability = match_result;
			matched_template = templates[j];
			template_index = j;
			results[index] = j + 1;
		}
	}

	cout << "] [MATCHED TEMPLATE " + to_string(template_index) + "]";
	return matched_template;
}

