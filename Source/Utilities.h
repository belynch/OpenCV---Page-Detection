/*
 * This code is provided as part of "A Practical Introduction to Computer Vision with OpenCV"
 * by Kenneth Dawson-Howe © Wiley & Sons Inc. 2014.  All rights reserved.
 */
#include "opencv2/core.hpp"
#include "Histograms.hpp" 

void writeText( cv::Mat image, char* text, int row, int column, cv::Scalar colour=-1.0, double scale=0.4, int thickness=1 );
cv::Mat JoinImagesHorizontally( cv::Mat& image1, char* name1, cv::Mat& image2, char* name2, int spacing=0, cv::Scalar colour=-1.0 );
cv::Mat JoinImagesVertically( cv::Mat& image1, char* name1, cv::Mat& image2, char* name2, int spacing=0, cv::Scalar colour=-1.0 );

/*
 * Performs back projection on a given image using a sample. 
 * @param image: the given image
 * @param sample_image: the image used as a sample
 * @return back_projection_probabilities: greyscale probability image
 */
cv::Mat backProjection(cv::Mat& image, cv::Mat& sample_image);

/*
 * Wrapper for the OpenCV threshold function
 * @param gray_image: the given image to threshold
 * @param currentThreshold: the threshold value (0-255)
 * @return binary_image: the thresholded image
 */
cv::Mat manualThreshold(cv::Mat& gray_image, int currentThreshold);

/*
 * Wrapper for the OpenCV morphologyEx function
 * @param binary_image: the given image
 * @param morph_type: the operation type (MORPH_OPEN, MORPH_CLOSE, MORPH_DILATE, MORPH_ERODE)
 * @param iterations: the number of times the operation is applied
 * @return result: the resultant image
 */
cv::Mat morphology(cv::Mat& binary_image, int morph_type, int iterations);

/*
 * Detects the 4 corners of a rotated book page which has blue marks
 * @param image: the given image
 * @param topLeft, topRight, bottomLeft, bottomRight:  results overwrite given references 
 */
void detectCorners(cv::Mat image, cv::Point& topLeft, cv::Point& topRight, cv::Point& bottomLeft, cv::Point& bottomRight);

/*
 * Draws circles on the corners of the marked page in the image
 * @param image: the given image
 * @param topLeft, topRight, bottomLeft, bottomRight:  4 corners of page
 * @return result: a copy of the given image with circles drawn
 */
cv::Mat addCircles(cv::Mat& image, cv::Point topLeft, cv::Point topRight, cv::Point bottomLeft, cv::Point bottomRight);

/*
 * Detects the 4 corners of a straight, non-perspective book page which has blue marks
 * @param image: the given image
 * @param topLeft, topRight, bottomLeft, bottomRight:  results overwrite given references 
 */
void detectTemplateCorners(cv::Mat image, cv::Point& topLeft, cv::Point& topRight, cv::Point& bottomLeft, cv::Point& bottomRight);

/*
 * Wrapper for the OpenCV warpPerspective function
 * @param source: the image to transform
 * @param templateImage: 4 corners of page
 * @param src_points: 4 corners of the source image
 * @param dst_points: where the 4 corners of result should be
 * @return output: the transformed image
 */
cv::Mat perspectiveTransformation(cv::Mat source, cv::Mat templateImage, cv::Point2f src_points[], cv::Point2f dst_points[]);

/*
 * Wrapper for the OpenCV matchTemplate function
 * @param src: the image to be matched
 * @param template_img: the template image
 * @return maxVal: the probability of match
 */
double templateMatching(cv::Mat src, cv::Mat template_img);

/*
 * Wrapper for the OpenCV Canny function
 * @param src: the BGR image to apply edge detection to
 * @return draw: a drawable edge image
 */
cv::Mat cannyEdgeDetection(cv::Mat src);