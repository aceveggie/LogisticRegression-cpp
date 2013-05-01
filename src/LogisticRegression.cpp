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
	assert(Labels.rows == Data.rows);
	this->n_classes = LogisticRegression::LR::get_label_map(Labels).size();
	cout<<"total classes found: "<<this->n_classes<<endl;
	assert(this->n_classes>=2);

}

cv::Mat LogisticRegression::LR::train(Mat DataI, Mat LabelsI, vector<int> unique_classes)
{
	Mat Data;
	Mat Labels;
	Mat Thetas;
	Mat Init_Theta = Mat::zeros(DataI.cols, 1, CV_64F);
	Mat LLabels = Mat::zeros(LabelsI.rows, LabelsI.cols, CV_8U);

	std::vector<int> cunique_classes;

	for(int i=0;i<unique_classes.size();i++)
	{
		int cclass = unique_classes.at(i);
		for(int j=0;j<LabelsI.rows;j++)
		{
			if(LabelsI.at<int>(j, 0) == cclass)
				LLabels.at<int>(j, 0) = i;
		}
		// to hold new class label information
		cunique_classes.push_back(i);
	}
	
	cout<<"LLabels"<<LLabels<<endl;
	

	if(this->n_classes == 2)
	{
		DataI.convertTo(Data, CV_64F);
		LLabels.convertTo(Labels, CV_64F);
		cout<<"Old cost: "<<LogisticRegression::LR::compute_cost(Data, Labels, Init_Theta)<<endl;
		cout<<"Old theta: "<<Init_Theta<<endl;
		Mat NewTheta = LogisticRegression::LR::compute_gradient(Data, Labels, Init_Theta);
		cout<<"New cost: "<<LogisticRegression::LR::compute_cost(Data, Labels, NewTheta)<<endl;
		cout<<"New Theta: "<<NewTheta<<endl;
		cout<<"Old theta: "<<Init_Theta<<endl;

	}

	
	
	
	return Thetas;

}

cv::Mat LogisticRegression::LR::predict(Mat Data, cv::Mat Thetas)
{
	// returns a class of the predicted class
	// class names can be 1,2,3,4, .... etc
	assert(Thetas.rows > 0);

	int classified_class = 0;
	double minVal;
	double maxVal;

	Point minLoc;
	Point maxLoc;
    Point matchLoc;
	
	vector<int> m_count;
	Labels = cv::Mat(Data.rows, 1, Data.type());
	cv::Mat TempPred;
	//cv::Mat MPred;
	cv::Mat MPred = Mat::zeros(Data.rows, Thetas.rows, Data.type());
	//cv::Mat Pred = Mat(Labels);
	
	

	if(Thetas.rows == 1)
	{
		TempPred = LogisticRegression::LR::calc_sigmoid(Data*Thetas.col(0));
		
		TempPred.at<int>(0,0) = floor(TempPred.at<int>(0, 0) + 0.5);
	}
	else
	{

		cout<<Thetas.cols<<endl;
		for(int i = 0;i<Thetas.cols;i++)
		{
			cout<<"Thetas "<<i<<Thetas.col(i)<<endl;
			TempPred = LogisticRegression::LR::calc_sigmoid(Data * Thetas.col(i));
			//assert(TempPred.cols ==1);
			//cout<<"assinging"<<TempPred.rows<<", "<<TempPred.cols<<endl;
			//MPred(Range::all(), Range(i,i+1)) = TempPred;
			//MPred.col(i) = TempPred;
			// cout<<"before"<<endl;
			// cout<<MPred<<endl;
			cv::vconcat(TempPred, MPred.col(i));
			// cout<<"after"<<endl;
			// cout<<MPred<<endl;
		}
		
		for(int i = 0;i<Data.rows;i++)
		{
			TempPred = MPred.row(i);
			//cout<<"TempPred"<<TempPred<<endl;
			minMaxLoc( TempPred, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
			//cout<<"predicted: "<<maxLoc.y+1<<endl;
			Labels.at<int>(i,0) = maxLoc.y+1;
		}
	}
	return Labels;
}

cv::Mat LogisticRegression::LR::calc_sigmoid(Mat Data)
{
	cv::Mat Dest;
	cv::exp(-Data, Dest);
	// cout<<"sigmoid of Data: "<<Data<<endl;
	// cout<<"sigmoid of Dest: "<<Dest<<endl;
	// cout<<"1.0/(1.0+Dest): "<<1.0/(1.0+Dest)<<endl;
	return 1.0/(1.0+Dest);
}


double LogisticRegression::LR::compute_cost(Mat Data, Mat Labels, Mat Init_Theta)
{
	//cout<<"received "<<Init_Theta<<endl;
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

	Mat D2 = 1 - LogisticRegression::LR::calc_sigmoid(Data * Init_Theta);
	cv::log(D2, D2);
	cv::multiply(D2, 1-Labels, D2);
	

	cost = (-1.0/m) * (cv::sum(D1)[0] + cv::sum(D2)[0]);
	cost = cost + rparameter;

	return cost;
}

cv::Mat LogisticRegression::LR::compute_gradient(Mat Data, Mat Labels, Mat Init_Theta)
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

	if(this->regularized == true)
	{
		llambda = 1;
	}
	
	for(int i = 0;i<this->num_iters;i++)
	{
		ccost = LogisticRegression::LR::compute_cost(Data, Labels, PTheta);
		if(this->debug == true)
		{
			cout<<"iter: "<<i<<endl;
			cout<<"cost: "<<ccost<<endl;
		}

		B = LogisticRegression::LR::calc_sigmoid((Data*PTheta) - Labels);
		A = ((double)1/m) * Data.t();
		//cout<<A<<endl;
		Gradient = A * B;
		//cout<<Gradient<<endl;

		A = LogisticRegression::LR::calc_sigmoid(Data*PTheta) - Labels;
		B = Data(Range::all(), Range(0,1)).reshape((Data.rows,1));
		
		cv::multiply(A, B, AB, 1);
		// cout<<"AB"<<AB<<endl;
		// cout<<"sum: "<<sum(AB)[0];
		// cout<<"---"<<endl;
		// for(int i =0;i<PTheta.rows;i++)
		// {
		// 	cout<<PTheta.at<long double>(i,0)<<endl;
		// }
		// cout<<PTheta.rows<<", "<<PTheta.cols<<endl;
		// cout<<"---"<<endl;
		// exit(0);
		Gradient.row(0) = ((float)1/m) * sum(AB)[0];
		// cout<<"changing vlaues: "<<((float)1/m) * sum(AB)[0];

		B = Data(Range::all(), Range(1,n));
		// cout<<"Sub B\n"<<B<<endl;
		// cout<<Gradient.rows<<", "<<Gradient.cols<<endl;
		// cout<<PTheta.rows<<", "<<PTheta.cols<<endl;
		// cout<<"---"<<endl;

		for(int i = 1;i<Gradient.rows;i++)
		{
			B = Data(Range::all(), Range(i,i+1));
			
			cv::multiply(A, B, AB, 1);
			Gradient.row(i) = (1.0/m)*cv::sum(AB)[0] + (llambda/m) * PTheta.row(i);
			//cout<<(1.0/m)*cv::sum(AB)[0] + (llambda/m) * PTheta.row(i)<<endl;
		}

		PTheta = PTheta - ((long double)this->alpha/m)*Gradient;
		
	}
	//cout<<Data*PTheta<<endl;
	return PTheta;

}

std::map<int, int> LogisticRegression::LR::get_label_map(Mat Labels)
{
	std::map<int, int> label_map;
	for(int i = 0;i<Labels.rows;i++)
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