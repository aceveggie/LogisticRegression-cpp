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

#include <iostream>
#include <assert.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "LogisticRegression.hpp"

using namespace cv;
using namespace std;


LogisticRegression::CvLR_TrainParams::CvLR_TrainParams()
{

}

LogisticRegression::CvLR_TrainParams::CvLR_TrainParams(double _alpha, int _num_iters, int _normalization, bool _debug, bool _regularized): alpha(_alpha), num_iters(_num_iters), normalization(_normalization), debug(_debug), regularized(_regularized)
{

}
LogisticRegression::CvLR_TrainParams::~CvLR_TrainParams()
{

}

cv::Mat LogisticRegression::CvLR::train(Mat DataI, Mat LabelsI, CvLR_TrainParams params)
{
	// cout<<"params.alpha = "<<params.alpha<<endl;
	// cout<<"params.num_iters = "<<params.num_iters<<endl;
	// cout<<"params.normalization = "<<params.normalization<<endl;
	// cout<<"params.debug = "<<params.debug<<endl;
	// cout<<"params.regularized = "<<params.regularized<<endl;
	// exit(0);

	int num_classes = LogisticRegression::CvLR::get_label_map(LabelsI).size();
	vconcat(Mat(DataI.rows, 1, DataI.type(), Scalar::all(1.0)), DataI.col(0));
	CV_Assert(LabelsI.rows == DataI.rows);
	
	CV_Assert(num_classes>=2);

	Mat Data;
	Mat Labels;

	//vector<int> unique_classes = LogisticRegression::CvLR::get_label_list( LogisticRegression::CvLR::get_label_map(LabelsI));
	
	Mat Thetas = Mat::zeros(num_classes, DataI.cols, CV_64F);
	Mat Init_Theta = Mat::zeros(DataI.cols, 1, CV_64F);
	Mat LLabels = LogisticRegression::CvLR::remap_labels(LabelsI, this->forward_mapper);
	Mat NewLocalLabels;
	
	int ii=0;
	
	if(num_classes == 2)
	{
		DataI.convertTo(Data, CV_64F);
		
		LLabels.convertTo(Labels, CV_64F);
		
		Mat NewTheta = LogisticRegression::CvLR::compute_gradient(Data, Labels, Init_Theta, params);
		
		Thetas = NewTheta.t();
	}

	else
	{
		/* take each class and rename classes you will get a theta per class
		as in multi class class scenario, we will have n thetas for n classes */
		ii = 0;

  		for(map<int,int>::iterator it = this->forward_mapper.begin(); it != this->forward_mapper.end(); ++it) 
		{
			NewLocalLabels = (LLabels == it->second)/255;
			cout<<"processing class "<<it->second<<endl;
			DataI.convertTo(Data, CV_64F);
			NewLocalLabels.convertTo(Labels, CV_64F);
			cout<<"initial theta: "<<Init_Theta<<endl;
			Mat NewTheta = LogisticRegression::CvLR::compute_gradient(Data, Labels, Init_Theta, params);
			cout<<"learnt theta: "<<NewTheta<<endl;
			hconcat(NewTheta.t(), Thetas.row(ii));
			ii += 1;
		}

	}

	return Thetas;
}

cv::Mat LogisticRegression::CvLR::predict(Mat Data, cv::Mat Thetas)
{
	/* returns a class of the predicted class
	 class names can be 1,2,3,4, .... etc */

	// add a column of ones
	vconcat(Mat(Data.rows, 1, Data.type(), Scalar::all(1.0)), Data.col(0));
	
	CV_Assert(Thetas.rows > 0);

	int classified_class = 0;
	double minVal;
	double maxVal;

	Point minLoc;
	Point maxLoc;
    Point matchLoc;
	
		
	cv::Mat Labels;
	cv::Mat CLabels;
	cv::Mat TempPred;
	
	cv::Mat MPred = Mat::zeros(Data.rows, Thetas.rows, Data.type());
	
	if(Thetas.rows == 1)
	{
		TempPred = LogisticRegression::CvLR::calc_sigmoid(Data*Thetas.t());
		CV_Assert(TempPred.cols==1);
		TempPred = (TempPred>0.5)/255;
		
		TempPred.convertTo(CLabels, CV_32S);
		CLabels = LogisticRegression::CvLR::remap_labels(CLabels, this->reverse_mapper);
		
	}
	else
	{
		for(int i = 0;i<Thetas.rows;i++)
		{
			TempPred = LogisticRegression::CvLR::calc_sigmoid(Data * Thetas.row(i).t());
			cv::vconcat(TempPred, MPred.col(i));
		}

		
		for(int i = 0;i<MPred.rows;i++)
		{
			TempPred = MPred.row(i);
	
			minMaxLoc( TempPred, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
			Labels.push_back(maxLoc.x);
		}
	
		Labels.convertTo(CLabels, CV_32S);
		CLabels = LogisticRegression::CvLR::remap_labels(CLabels, this->reverse_mapper);
		
	}
	return CLabels;
}

cv::Mat LogisticRegression::CvLR::calc_sigmoid(Mat Data)
{
	cv::Mat Dest;
	cv::exp(-Data, Dest);
	return 1.0/(1.0+Dest);
}


//double LogisticRegression::CvLR::compute_cost(Mat Data, Mat Labels, Mat Init_Theta)
double LogisticRegression::CvLR::compute_cost(Mat Data, Mat Labels, Mat Init_Theta, CvLR_TrainParams params)
{
	
	int llambda = 0;
	int m;
	int n;

	double cost = 0;
	double rparameter = 0;

	cv::Mat Gradient;
	cv::Mat Theta2;
	cv::Mat Theta2Theta2;
	
	m = Data.rows;
	n = Data.cols;
	
	
	Gradient = Mat::zeros( Init_Theta.rows, Init_Theta.cols, Init_Theta.type());
	
	Theta2 = Init_Theta(Range(1, n), Range::all());
		
	cv::multiply(Theta2, Theta2, Theta2Theta2, 1);
	
	if(params.regularized == true)
	{
		llambda = 1.0;
	}

	if(params.normalization == CvLR::REG_L1)
	{
		rparameter = (llambda/(2*m)) * cv::sum(Theta2)[0];
	}
	else
	{
		// assuming it to be L2 by default
		rparameter = (llambda/(2*m)) * cv::sum(Theta2Theta2)[0];
	}


	Mat D1 = LogisticRegression::CvLR::calc_sigmoid(Data* Init_Theta);
	

	cv::log(D1, D1);
	cv::multiply(D1, Labels, D1);

	Mat D2 = 1 - LogisticRegression::CvLR::calc_sigmoid(Data * Init_Theta);
	cv::log(D2, D2);
	cv::multiply(D2, 1-Labels, D2);
	
	cost = (-1.0/m) * (cv::sum(D1)[0] + cv::sum(D2)[0]);
	cost = cost + rparameter;

	return cost;
}

cv::Mat LogisticRegression::CvLR::compute_gradient(Mat Data, Mat Labels, Mat Init_Theta, CvLR_TrainParams params)
{
	int llambda = 0;
	long double ccost;
	int m, n;

	cv::Mat A;
	cv::Mat B;
	cv::Mat AB;
	cv::Mat Gradient;
	cv::Mat PTheta = Init_Theta.clone();
	
	
	m = Data.rows;
	n = Data.cols;

	if(params.regularized == true)
	{
		llambda = 1;
	}
	
	for(int i = 0;i<params.num_iters;i++)
	{
		ccost = LogisticRegression::CvLR::compute_cost(Data, Labels, PTheta, params);

		if(params.debug == true && i%(params.num_iters/2)==0)
		{	
			cout<<"iter: "<<i<<endl;
			cout<<"cost: "<<ccost<<endl;
		}

		B = LogisticRegression::CvLR::calc_sigmoid((Data*PTheta) - Labels);
		A = ((double)1/m) * Data.t();
		
		Gradient = A * B;
		

		A = LogisticRegression::CvLR::calc_sigmoid(Data*PTheta) - Labels;
		B = Data(Range::all(), Range(0,1)).reshape((Data.rows,1));
		
		cv::multiply(A, B, AB, 1);
		
		Gradient.row(0) = ((float)1/m) * sum(AB)[0];
		

		B = Data(Range::all(), Range(1,n));
		

		for(int i = 1;i<Gradient.rows;i++)
		{
			B = Data(Range::all(), Range(i,i+1));
			
			cv::multiply(A, B, AB, 1);
			Gradient.row(i) = (1.0/m)*cv::sum(AB)[0] + (llambda/m) * PTheta.row(i);
			
		}
		// cout<<"old ptheta"<<PTheta<<endl;
		// cout<<"params.alpha: "<<params.alpha<<endl;
		// cout<<"m: "<<m<<endl;
		// cout<<"(double)params.alpha/m: "<<(double)params.alpha/m<<endl;
		PTheta = PTheta - ((double)params.alpha/m)*Gradient;
		// cout<<"changed PTheta"<<PTheta<<endl;
		
	}
	
	return PTheta;

}

std::map<int, int> LogisticRegression::CvLR::get_label_map(Mat Labels)
{
	// this function creates two maps to map user defined labels to program friendsly labels
	// two ways.
	CV_Assert(Labels.type() == CV_32S);

	std::map<int, int> forward_mapper;
	std::map<int, int> reverse_mapper;
	

	for(int i = 0;i<Labels.rows;i++)
	{
		forward_mapper[Labels.at<int>(i)] += 1;
	}
	
	int ii = 0;

	for(map<int,int>::iterator it = forward_mapper.begin(); it != forward_mapper.end(); ++it) 
	{
	 	forward_mapper[it->first] = ii;
	 	ii += 1;
  	}
  	  	
  	for(map<int,int>::iterator it = forward_mapper.begin(); it != forward_mapper.end(); ++it) 
	{
	 	reverse_mapper[it->second] = it->first;
	}

	this->forward_mapper = forward_mapper;
	this->reverse_mapper = reverse_mapper;
	
	return forward_mapper;
}

vector<int> LogisticRegression::CvLR::get_label_list(std::map<int, int> lmap)
{
	std::vector<int> v;
	for(map<int,int>::iterator it = lmap.begin(); it != lmap.end(); ++it) 
	{
	 	v.push_back(it->first);
  	}	
	return v;
}

cv::Mat LogisticRegression::CvLR::remap_labels(Mat Labels, std::map<int, int> lmap)
{
	cv::Mat NewLabels = Mat::zeros(Labels.rows, Labels.cols, Labels.type());
	for(int i =0;i<Labels.rows;i++)
	{
		NewLabels.at<int>(i,0) = lmap[Labels.at<int>(i,0)];
	}
	return NewLabels;
}