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

vector<cv::Mat> LogisticRegression::LR::train(Mat Data, Mat Labels, vector<int> unique_classes)
{
	int num_iters = this->num_iters;
	int m = Data.rows;
	int n = Data.cols;
	double cost;
	bool status = false;
	

	cv::Mat LLabels = Mat(Labels.rows, Labels.cols, Labels.type(), Scalar::all(0));
	cv::Mat TempTheta;
	cv::Mat NewTheta;
	cv::Mat TLabels;

	vector<cv::Mat> InitThetas;
	vector<cv::Mat> Thetas;
	std::vector<int> cunique_classes;

	// rename user defined labels for more simplicity starting from 0, 1, 2, ...
	for(int i=0;i<unique_classes.size();i++)
	{
		int cclass = unique_classes[i];
		for(int j=0;j<Labels.rows;j++)
		{
			if(Labels.at<int>(j, 0)==cclass)
				LLabels.at<int>(j, 0) = i;

		}
		// to hold new class label information
		cunique_classes.push_back(i);
	}
	

	if(this->n_classes==2)
	{
		TempTheta == Mat::zeros(1, Data.cols, CV_32FC1);
		NewTheta = LogisticRegression::LR::compute_gradient(Data, LLabels, TempTheta);
		cost = LogisticRegression::LR::compute_cost(Data, LLabels, NewTheta);
		InitThetas.push_back(NewTheta);
		status = true;
	}
	else
	{
		// we need to get a theta per class
		// for a single class
		for(int i =0;i<this->n_classes;i++)
		{
			TempTheta == Mat::zeros(1, Data.cols, CV_32FC1);
			InitThetas.push_back(TempTheta);
		}
		for(int i =0 ;i<this->n_classes;i++)
		{
			// take each class
			// make this class 1 and rest of them zero
			
			int cclass = cunique_classes[i];
			TLabels = Mat::zeros(Labels.rows, Labels.cols, Labels.type());
			for(int j = 0;j<TLabels.rows;j++)
			{
				if(cunique_classes[i] == LLabels.at<int>(j,0))
				{
					TLabels.at<int>(j,0) = 1;
				}
			}

			// for current class TLabels has value of 1 and rest of them as zero
			TempTheta = InitThetas[i];
			NewTheta = LogisticRegression::LR::compute_gradient(Data, TLabels, TempTheta);
			ccost =LogisticRegression::LR::compute_cost(Data, TLabels, TempTheta);

			Thetas.push_back(NewTheta);
		}


	}

	return Thetas;
		
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
	return classified_class;
}

cv::Mat LogisticRegression::LR::calc_sigmoid(Mat Data)
{
	cv::Mat Dest;
	cv::exp(-Data, Dest);
	return 1.0/(1.0+Dest);
}


double LogisticRegression::LR::compute_cost(Mat Data, Mat Labels, Mat Init_Theta)
{
	int llambda = 0;
	int m;
	int n;

	double cost = 0;
	double rparameter = 0

	cv::Mat Gradient;
	cv::Mat Theta2;
	cv::Mat Theta2Theta2;
	
	m = Data.rows;
	n = Data.cols;

	Gradient = Mat::zeros( Init_Theta.rows, Init_Theta.cols, Init_Theta.type());
	Theta2 = Init_Theta(Range(1, n), Range::all());
	cv::multiply(Theta2, Theta2, Theta2Theta2, 1);

	if(this->regularized == true)
	{
		llambda = 1.0;
	}

	if(this->normalization_mode.compare("L1"))
	{
		rparameter = (llambda/(2*m)) * cv::sum(Theta2)[0];	
	}
	else
	{
		// assuming it to be L2 by default
		rparameter = (llambda/(2*m)) * cv::sum(Theta2Theta2)[0];
	}

	
	Mat D1 = LogisticRegression::LR::calc_sigmoid(Data* Init_Theta);
	cv::log(D1, D1);
	cv::multiply(D1, Labels, D1);

	Mat D2 = 1 - LogisticRegression::LR::calc_sigmoid(np.dot(Data * Init_Theta));
	cv::log(D2, D2);
	cv::multiply(D2, 1-Labels, D2);
	
	cost = (-1.0/m) * (cv::sum(D1)[0] + cv::sum(D2)[0]);
	cost = cost + rparameter;

	// J = (-1.0/ m) * ( np.sum( np.log(self.sigmoidCalc( np.dot(data, init_theta))) * labels + ( np.log ( 1 - self.sigmoidCalc(np.dot(data, init_theta)) ) * ( 1 - labels ) ) ) )
	// J = J + regularized_parameter

	return cost;



}
void LogisticRegression::LR::compute_gradient(Mat Data, Mat Labels, Mat Init_Theta)
{
	cv::Mat A;
	cv::Mat B;
	cv::Mat Gradient;
	cv::Mat AB;

	int alpha = this->alpha;
	int num_iters = this->num_iters;
	int llambda = 0;
	int num_samples = Labels.rows;
	long double cost = 0;
	int m;
	int n;
	m = Data.rows;
	n = Data.cols;


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
	B = Data(Range::all(), 0);

	cv::multiply(A, B, AB, 1);
	
	Gradient.at<double>(0,0) = (1/m) * cv::sum(AB)[0];	
	B = Data(Range::all(), Range(1,n));

	for(int i = 1;i<Gradient.rows;i++)
	{
		// B = (data[:,i].reshape((data[:,i].shape[0],1)))
		// grad[i] = (1/m)*np.sum(A*B) + ((llambda/m)*init_theta[i])

		B = Data(Range:all(),i).reshape((data(Range:all(),i).rows,1));
		cv::multiply(A, B, AB, 1);
		Gradient.at<double>(0,i) = (1.0/m)*cv::sum(AB)[0] + (llambda/m) * Init_Theta.at<double>(0, i);
	}

	Init_Theta = Init_Theta - ((alpha/m)*Gradient);
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

vector<int> LogisticRegression::LR::get_label_list(std::map<int, int> lmap)
{
	std::vector<int> v;

	for(map<int,int>::iterator it = lmap.begin(); it != lmap.end(); ++it) 
	{
	 	v.push_back(it->first);
	  	cout << it->first << "\n";
  	}
	
	return v;
}