#include "opencv2/core.hpp"

class Page
{
public:
	int width;
	int height;
	cv::Point2f corners[4];

	Page(){
		width = 0;
		height = 0;
	}
};