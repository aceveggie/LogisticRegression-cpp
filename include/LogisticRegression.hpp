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
#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>



using namespace cv;


namespace LogisticRegression 
{

	using namespace std;
	using namespace cv;
	
	struct CvLR_TrainParams
	{
		CvLR_TrainParams();
	    CvLR_TrainParams(double alpha, int num_iters, int normalization, bool debug, bool regularized);
	    ~CvLR_TrainParams();

	    double alpha;
	    bool regularized;
	    int num_iters;
	    int normalization;
	    bool debug;
	    
	};

	/* Logistic Regression */
	// class CvLR : public CvStatModel
	class CvLR
	{

		public:
			enum { REG_L1=0, REG_L2 = 1};

			CvLR()
			{
			}
			

			//LR(Mat Data, Mat Labels)
			CvLR(const Mat& Data, const Mat& Labels, const CvLR_TrainParams& params)
			{
				train(Data, Labels, params);
			}

			~CvLR()
			{
			}
			// for SVM bool train(const Mat& trainData, const Mat& responses, const Mat& varIdx=Mat(), const Mat& sampleIdx=Mat(), CvSVMParams params=CvSVMParams() )¶

			bool train(const Mat& DataI, const Mat& LabelsI, const CvLR_TrainParams& params);
			float predict(const Mat& Data, cv::Mat& PredLabs);
			float predict(const Mat& Data);
			
		protected:
			std::map<int, int> forward_mapper;
			std::map<int, int> reverse_mapper;
			cv::Mat learntThetas;

			CvLR_TrainParams params;

			bool set_default_params();
			cv::Mat calc_sigmoid(const Mat& Data);
			double compute_cost(const Mat& Data, const Mat& Labels, const Mat& Init_Theta, const CvLR_TrainParams& params);
			cv::Mat compute_batch_gradient(const Mat& Data, const Mat& Labels, const Mat& Init_Theta, const CvLR_TrainParams& params);
			std::map<int, int> get_label_map(const Mat& Labels);
			bool set_label_map(const Mat& Labels);
			cv::Mat remap_labels(const Mat& Labels, const std::map<int, int> lmap);
			void clear();
			void read();
			void write();
	};
/* end of namespace */
}
#endif