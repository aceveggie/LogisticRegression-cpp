/*
This file features a module to perform Logistic Regression in OpenCV
Author: Rahul Kavi
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


	/* Perform Linear Discriminant Analysis */
	class LR 
	{

		private:

			Mat Data;
			Mat Labels;
			vector<cv::Mat> Weights;
			float alpha;
			int n_classes;
			int num_iters;
			string normalization_mode;
			bool debug;
			bool regularized;


		public:

			LR(Mat Data, Mat Labels, int num_iters = 100,bool regularized = false,bool debug = false, float alpha = 1,string normalization_mode = "L2")
			{
				this->Data = Data;
				this->Labels = Labels;
				this->alpha = alpha;
				this->num_iters = num_iters;
				this->normalization_mode = normalization_mode;
				this->debug = debug;
				this->regularized = regularized;
				this->init(this->Data, this->Labels);
			}

			~LR()
			{
	
			}

			// void init(Mat Data, Mat Labels);
			// std::vector<cv::Mat> train(Mat Data, Mat Labels, vector<int> unique_classes);
			// void predict(Mat Data, vector<cv::Mat> Thetas);
			// void calc_sigmoid(Mat Data);
			// double compute_cost(Mat Data, Mat Labels, Mat Init_Theta);
			// cv::Mat compute_gradient(Mat Data, Mat Labels, Mat Init_Theta);
			// void get_label_map(Mat Labels);

			void init(Mat Data, Mat Labels);
			cv::Mat train(Mat Data, Mat Labels, vector<int> unique_classes);
			cv::Mat predict(Mat Data, Mat Thetas);
			cv::Mat calc_sigmoid(Mat Data);
			double compute_cost(Mat Data, Mat Labels, Mat Init_Theta);
			cv::Mat compute_gradient(Mat Data, Mat Labels, Mat Init_Theta);
			std::map<int, int> get_label_map(Mat Labels);
			vector<int> get_label_list(std::map<int, int> lmap);

	};
/* end of namespace */
}
#endif