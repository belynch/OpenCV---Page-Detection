/*
 * This code is provided as part of "A Practical Introduction to Computer Vision with OpenCV"
 * by Kenneth Dawson-Howe © Wiley & Sons Inc. 2014.  All rights reserved.
 */
#include "Utilities.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

void writeText( Mat image, char* text, int row, int column, Scalar passed_colour, double scale, int thickness )
{
	Scalar colour( 0, 0, 255);
	Point location( column, row );
	putText( image, text, location, FONT_HERSHEY_SIMPLEX, scale, (passed_colour.val[0] == -1.0) ? colour : passed_colour, thickness );
}

Mat JoinImagesHorizontally( Mat& image1, char* name1, Mat& image2, char* name2, int spacing, Scalar passed_colour/*=-1.0*/ )
{
	Mat result( (image1.rows > image2.rows) ? image1.rows : image2.rows,
		        image1.cols + image2.cols + spacing,
				image1.type() );
	result.setTo(Scalar(255,255,255));
    Mat imageROI;
	imageROI = result(cv::Rect(0,0,image1.cols,image1.rows));
	image1.copyTo(imageROI);
	if (spacing > 0)
	{
		imageROI = result(cv::Rect(image1.cols,0,spacing,image1.rows));
		imageROI.setTo(Scalar(255,255,255));
	}
	imageROI = result(cv::Rect(image1.cols+spacing,0,image2.cols,image2.rows));
	image2.copyTo(imageROI);
	writeText( result, name1, 13, 6, passed_colour );
	writeText( imageROI, name2, 13, 6, passed_colour );
	return result;
}

Mat JoinImagesVertically( Mat& image1, char* name1, Mat& image2, char* name2, int spacing, Scalar passed_colour/*=-1.0*/ )
{
	Mat result( image1.rows + image2.rows + spacing,
		        (image1.cols > image2.cols) ? image1.cols : image2.cols,
				image1.type() );
	result.setTo(Scalar(255,255,255));
	Mat imageROI;
	imageROI = result(cv::Rect(0,0,image1.cols,image1.rows));
	image1.copyTo(imageROI);
	if (spacing > 0)
	{
		imageROI = result(cv::Rect(0,image1.rows,image1.cols,spacing));
		imageROI.setTo(Scalar(255,255,255));
	}
	imageROI = result(cv::Rect(0,image1.rows+spacing,image2.cols,image2.rows));
	image2.copyTo(imageROI);
	writeText( result, name1, 13, 6, passed_colour );
	writeText( imageROI, name2, 13, 6, passed_colour );
	return result;
}


Mat backProjection(Mat& original_image, Mat& sample_image)
{
	Mat hls_image;
	cvtColor(sample_image, hls_image, CV_BGR2HLS);
	ColourHistogram histogram3D(hls_image,16);
	histogram3D.NormaliseHistogram();
	cvtColor(original_image, hls_image, CV_BGR2HLS);
	Mat back_projection_probabilities = histogram3D.BackProject(hls_image);
	return back_projection_probabilities;
}

Mat manualThreshold(Mat& gray_image, int currentThreshold)
{
	Mat  binary_image;
	int current_threshold = 160;
	int max_threshold = 255;
	threshold(gray_image, binary_image, currentThreshold, max_threshold, THRESH_BINARY);
	return binary_image;
}

Mat morphology(Mat& binary_image, int morph_type, int iterations)
{
	Mat result;
	morphologyEx(binary_image, result, morph_type, Mat(), Point(-1, -1), iterations);
	return result;
}

/*
 * Loops through the binary image and identifies the minimum and maximum x and y values, storing and 
 * returning the corresponding points.
 */
void detectCorners(Mat image, Point& topLeft, Point& topRight, Point& bottomLeft, Point& bottomRight) {
	Mat nonZeroCoordinates;
	findNonZero(image, nonZeroCoordinates);

	int maxX = 0;
	int maxY = 0;
	int minX = image.cols;
	int minY = image.rows;
	for (unsigned int j = 0; j < nonZeroCoordinates.total(); j++) {
		//need to find min and max xs & ys
		int curX = nonZeroCoordinates.at<Point>(j).x;
		int curY = nonZeroCoordinates.at<Point>(j).y;

		if (curX > maxX) {
			maxX = curX;
			topRight = nonZeroCoordinates.at<Point>(j);
		}
		if (curY > maxY) {
			maxY = curY;
			bottomRight = nonZeroCoordinates.at<Point>(j);
		}
		if (curX < minX) {
			minX = curX;
			bottomLeft = nonZeroCoordinates.at<Point>(j);
		}
		if (curY < minY) {
			minY = curY;
			topLeft = nonZeroCoordinates.at<Point>(j);
		}
	}
	
}

Mat addCircles(Mat& image, Point topLeft, Point topRight, Point bottomLeft, Point bottomRight) 
{
	Mat result;
	image.copyTo(result);
	int radius = 5;
	Scalar colour = Scalar(0, 255, 0);
	circle(result, topLeft, radius, colour, CV_FILLED);
	circle(result, topRight, radius, colour, CV_FILLED);
	circle(result, bottomLeft, radius, colour, CV_FILLED);
	circle(result, bottomRight, radius, colour, CV_FILLED);
	return result;
}

void detectTemplateCorners(Mat image, Point& topLeft, Point& topRight, Point& bottomLeft, Point& bottomRight) {
	Mat nonZeroCoordinates;
	findNonZero(image, nonZeroCoordinates);

	int maxX = 0;
	int maxY = 0;
	int minX = image.cols;
	int minY = image.rows;
	for (unsigned int j = 0; j < nonZeroCoordinates.total(); j++) {
		//need to find min and max xs & ys
		int curX = nonZeroCoordinates.at<Point>(j).x;
		int curY = nonZeroCoordinates.at<Point>(j).y;

		if (curX > maxX) {
			maxX = curX;
		}
		if (curY > maxY) {
			maxY = curY;
		}
		if (curX < minX) {
			minX = curX;
		}
		if (curY < minY) {
			minY = curY;
		}

		topLeft = Point(minX, maxY);
		topRight = Point(maxX, maxY);
		bottomLeft = Point(minX, minY);
		bottomRight = Point(maxX, minY);
	}
}

Mat perspectiveTransformation(Mat source, Mat templateImage, Point2f src_points[], Point2f dst_points[]) 
{
	Mat perspective_matrix(3, 3, CV_32FC1), output;
	Mat lambda;

	lambda = getPerspectiveTransform(src_points, dst_points);
	warpPerspective(source, output, lambda, source.size());
	return output;
}

/*
 * Matches an image against a template using the CV_TM_CCORR_NORMED match method. The better the match the higher
 * the resultant value for this method and so maxVal is used.
 */
double templateMatching(Mat src, Mat template_img)
{
	int match_method = CV_TM_CCORR_NORMED;
	Mat result;
	// Create the result matrix
	int result_cols =  src.cols - template_img.cols + 1;
	int result_rows = src.rows - template_img.rows + 1;
	result.create( result_rows, result_cols, CV_32FC1 );
	// Do the Matching 
	matchTemplate( src, template_img, result, match_method );
	// Localizing the best match with minMaxLoc
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;
	minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
	matchLoc = maxLoc;

	return maxVal;
}

Mat cannyEdgeDetection(Mat src)
{
	Mat gray, edge, draw;
	cvtColor(src, gray, CV_BGR2GRAY);
	Canny( gray, edge, 50, 150, 3);
	edge.convertTo(draw, CV_8U);
	return draw;
}