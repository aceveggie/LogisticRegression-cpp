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
	this->n_classes = LogisticRegression::LR::get_label_map(Mat Labels).size();
	assert(this->n_classes>=2);

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
	int num_iters = this->num_iters;
	int m = Data.rows;
	int n = Data.cols;
	// local labels for one vs rest classification
	Mat LLabels = Mat(Labels.rows, Labels.cols, Labels.type(), Scalar::all(0));
		
}

int LogisticRegression::LR::predict(Mat Data, vector<cv::Mat> Thetas)
{
	// returns a class of the predicted class
	// class names can be 1,2,3,4, .... etc
	assert(Thetas.size()>0);

	int classified_class = 0;
	
	vector<int> m_count;

	Mat Pred;

	if(Thetas.size()==1)
	{
		Pred = LogisticRegression::LR::calc_sigmoid(Data*Thetas[i]);
		assert(Pred.rows ==1);
		assert(Pred.cols ==1);
		classified_class = floor(Pred.at<int>(0,0) + 0.5);
	}
	else
	{
		for(int i = 0;i<Thetas.size();i++)
		{
			Pred = LogisticRegression::LR::calc_sigmoid(Data * Thetas[i]);
			assert(Pred.rows ==1);
			assert(Pred.cols ==1);
			m_count.push_back(Pred.at<int>(0,0));
		}
		// class indices start from 1
		classified_class = *( std::max_element( v2.begin(), v2.end() ) ) + 1;
	}
}

cv::Mat LogisticRegression::LR::calc_sigmoid(Mat Data)
{
	cv::Mat Dest;
	cv::exp(-Data, Dest);
	return 1.0/(1.0+Dest);
}


void LogisticRegression::LR::compute_cost(Mat Data, Mat Labels, Mat Init_Theta)
{

}
void LogisticRegression::LR::compute_gradient(Mat Data, Mat Labels, Mat Init_Theta)
{
	cv::Mat A;
	cv::Mat B;
	cv::Mat Gradient;

	int alpha = this->alpha;
	int num_iters = this->num_iters;
	int llambda = 0;
	int num_samples = Labels.rows;
	long double cost = 0;


	if(this->regularized == true)
	{
		llambda = 1;
	}
	for(int i = 0;i<num_iters;i++)
	{
		cost = LogisticRegression::LR::compute_cost(Data, Labels, Init_Theta);
		if(this->debug == true)
		{
			cout<<"iteration: "<<i<<endl;
			cout<<"cost: "<<cost<<endl;
		}
	}
	B = LogisticRegression::LR::calc_sigmoid((Data*Init_Theta)-Labels);
	A = (1/num_samples)*Data.t();

	Gradient = A*B;


			
	A = LogisticRegression::LR::calc_sigmoid(Data*Init_Theta) - Labels;

	B = (data[:,range(1,n)])



}
std::map<int, int> LogisticRegression::LR::get_label_map(Mat Labels)
{
	std::map<int, int> label_map;
	for(int i = 0;i<Labels.row;i++)
	{
		label_map[Labels.at<int>(i)] += 1;
	}
	return label_map;
}