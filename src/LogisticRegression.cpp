/*
This file features a module to perform Logistic Regression in OpenCV
Author: Rahul Kavi
www.github.com/aceveggie
*/

#include <iostream>
#include <assert.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "LogisticRegression.hpp"

using namespace cv;
using namespace std;

void LogisticRegression::LR::init(Mat Data, Mat Labels)
{
	assert(labels.rows == Data.rows);
	std::map<string, int> label_map = LogisticRegression::LR::get_label_map(Mat Labels)
	
	//Mat m= Mat(4,3, CV_8UC1); uchar elem_m= m.at<uchar>(i,j); //access element mij, with i from 0 to rows-1 and j from 0 to cols-1. 

	/*
	Mat Data;
	Mat Labels;
	vector<cv::Mat> Weights;
	float alpha;
	int n_classes;
	int num_iters;
	string normalization_mode;
	bool debug;
	bool regularized;
	*/


}

void LogisticRegression::LR::train(Mat Data, Mat Labels, vector<int> unique_classes)
{

}

void LogisticRegression::LR::predict(Mat Data, vector<cv::Mat> Thetas)
{

}

void LogisticRegression::LR::calc_sigmoid((Mat Data)
{

}
void LogisticRegression::LR::compute_cost(Mat Data, Mat Labels, Mat init_theta)
{

}
void LogisticRegression::LR::compute_gradient(self,data, labels, init_theta)
{

}
void LogisticRegression::LR::get_label_map(Mat Labels)
{
	for(int i = 0;i<Labels.row)

}