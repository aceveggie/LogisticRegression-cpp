/*

This is a implementation of the Logistic Regression algorithm in C++ using OpenCV.

AUTHOR: RAHUL KAVI

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

#
# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Logistic Regression ALGORITHM

www.github.com/aceveggie

*/

#ifndef __LOGISTICREGRESSION_HPP__
#define __LOGISTICREGRESSION_HPP__


#include <iostream>
#include <assert.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>



using namespace cv;


namespace LogisticRegression 
{

	using namespace std;
	using namespace cv;
	
	struct CV_EXPORTS_W_MAP CvLR_TrainParams
	{
		CvLR_TrainParams();
	    
	    CvLR_TrainParams(double alpha, int num_iters, int normalization, bool debug, bool regularized);
	    
	    ~CvLR_TrainParams();

	    CV_PROP_RW double alpha;
	    CV_PROP_RW bool regularized;
	    CV_PROP_RW int num_iters;
	    CV_PROP_RW int normalization;
	    CV_PROP_RW bool debug;
	    
	};

	/* Logistic Regression */
	class CvLR 
	{

		public:
			enum { REG_L1=0, REG_L2 = 1};			
			map<int, int> forward_mapper;
			map<int, int> reverse_mapper;

		public:
			CvLR()
			{
			}

			//LR(Mat Data, Mat Labels)
			CvLR(Mat Data, Mat Labels, CvLR_TrainParams params)
			{
				cout<<"params.alpha = "<<params.alpha<<endl;
				cout<<"params.num_iters = "<<params.num_iters<<endl;
				cout<<"params.normalization = "<<params.normalization<<endl;
				cout<<"params.debug = "<<params.debug<<endl;
				cout<<"params.regularized = "<<params.regularized<<endl;
				//exit(0);
				train(Data, Labels, params);
			}

			~CvLR()
			{
			}

			cv::Mat train(Mat DataI, Mat LabelsI, CvLR_TrainParams params);
			cv::Mat predict(Mat Data, Mat Thetas);
			cv::Mat calc_sigmoid(Mat Data);
			double compute_cost(Mat Data, Mat Labels, Mat Init_Theta, CvLR_TrainParams params);
			cv::Mat compute_gradient(Mat Data, Mat Labels, Mat Init_Theta, CvLR_TrainParams params);
			std::map<int, int> get_label_map(Mat Labels);
			vector<int> get_label_list(std::map<int, int> lmap);
			cv::Mat remap_labels(Mat Labels, std::map<int, int> lmap);

	};
/* end of namespace */
}
#endif