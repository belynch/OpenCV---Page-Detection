/*
 * This code is provided as part of "A Practical Introduction to Computer Vision with OpenCV"
 * by Kenneth Dawson-Howe © Wiley & Sons Inc. 2014.  All rights reserved.
 */
#include "opencv2/core.hpp"

#define DEFAULT_MIN_SATURATION 25
#define DEFAULT_MIN_VALUE 25
#define DEFAULT_MAX_VALUE 230

class Histogram
{
protected:
	cv::Mat mImage;
	int mNumberChannels;
	int* mChannelNumbers;
	int* mNumberBins;
	float mChannelRange[2];
public:
	Histogram( cv::Mat image, int number_of_bins );
	virtual void ComputeHistogram() = 0;
	virtual void NormaliseHistogram() = 0;
	static void Draw1DHistogram( cv::MatND histograms[], int number_of_histograms, cv::Mat& display_image );
};

class ColourHistogram : public Histogram
{
private:
	cv::MatND mHistogram;
public:
	ColourHistogram( cv::Mat image, int number_of_bins );
	void ComputeHistogram();
	void NormaliseHistogram();
	cv::Mat BackProject( cv::Mat& image );
	cv::MatND getHistogram();
};



