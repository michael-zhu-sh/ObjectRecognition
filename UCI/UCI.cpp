/*
UCI Letter Recognition.
dataset url:http://archive.ics.uci.edu/ml/datasets/Letter+Recognition
Attribute Information:
1. lettr capital letter (26 values from A to Z) 
2. x-box horizontal position of box (integer) 
3. y-box vertical position of box (integer) 
4. width width of box (integer) 
5. high height of box (integer) 
6. onpix total # on pixels (integer) 
7. x-bar mean x of on pixels in box (integer) 
8. y-bar mean y of on pixels in box (integer) 
9. x2bar mean x variance (integer) 
10. y2bar mean y variance (integer) 
11. xybar mean x y correlation (integer) 
12. x2ybr mean of x * x * y (integer) 
13. xy2br mean of x * y * y (integer) 
14. x-ege mean edge count left to right (integer) 
15. xegvy correlation of x-ege with y (integer) 
16. y-ege mean edge count bottom to top (integer) 
17. yegvx correlation of y-ege with x (integer)

"The dataset consists of 20000 feature vectors along with the\n"
"responses - capital latin letters A..Z.\n"
"The first 16000 (10000 for boosting)) samples are used for training\n"
"and the remaining 4000 (10000 for boosting) - to test the classifier.\n"

使用	高斯核SVM,参数
	gamma=0.25,C=20.	
	错误率5.x%.
*/
#include "stdafx.h"
#include <fstream>
#include <iostream>
#include <string>

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;

const int VAR_COUNT = 16;
const int TRAINSET_SIZE= 16000;
const int TESTSET_SIZE = 4000;
float trainDatas[TRAINSET_SIZE][VAR_COUNT];
float testDatas[TESTSET_SIZE][VAR_COUNT];
int trainLabels[TRAINSET_SIZE];
char testLabels[TESTSET_SIZE];


void SplitString(const string &str, vector<string> &vars, const std::string pattern)
{
  string::size_type pos1, pos2;
  pos2 = str.find(pattern);
  pos1 = 0;
  while(std::string::npos != pos2)
  {
    vars.push_back(str.substr(pos1, pos2-pos1));
 
    pos1 = pos2 + pattern.size();
    pos2 = str.find(pattern, pos1);
  }
  if(pos1 != str.length())
    vars.push_back(str.substr(pos1));
}

/*
读取数据集到全局变量中。
*/
//int readDataset(Mat &trainMat, Mat &trainLabelMat, Mat &testMat, Mat &testLabelMat)	{
int readDataset()	{
	const string pattern(",");
	ifstream fin("C:/ImageDatabase/UCI/LetterRecognition/letter-recognition.data");//创建一个fstream文件流对象
	if (fin.eof())	{
		cerr<<"FAIL to open UCI dataset file."<<endl;
	}
    string line; //保存读入的每一行
	vector<string> vars;
	stringstream ss;
	int totalLineCnt=0,i,var;
	while(getline(fin, line))	{
		SplitString(line, vars, ",");

		if (totalLineCnt<TRAINSET_SIZE)	{
			trainLabels[totalLineCnt] = vars[0].at(0);	//vars[] is label, vars[1..16] is attribute.
			for(i=1; i!=VAR_COUNT+1; i++)	{
				ss<<vars[i];
				ss>>var;
				trainDatas[totalLineCnt][i-1] = static_cast<float>(var);
				ss.clear();
			}
		} else {
			testLabels[totalLineCnt-TRAINSET_SIZE] = vars[0].at(0);
			for(i=1; i!=VAR_COUNT+1; i++)	{
				ss<<vars[i];
				ss>>var;
				testDatas[totalLineCnt-TRAINSET_SIZE][i-1] = static_cast<float>(var);
				ss.clear();
			}
			//printf("test label is %c.\n", testLabels[totalLineCnt-TRAINSET_SIZE]);
		}

		vars.clear();
		totalLineCnt++;
	}
	cout<<"total lines:"<<totalLineCnt<<endl;
	fin.close();

	return  0;
}

int check()	{
	for(int i=0; i!=10; i++)	{
		cout<<"train data "<<i<<":";
		for(int j=0; j!=VAR_COUNT; j++)	{
			cout<<trainDatas[i][j]<<" ";
		}
		cout<<endl;
	}

	return 0;
}

int _tmain(int argc, _TCHAR* argv[])
{
	readDataset();

	Mat trainMat(TRAINSET_SIZE, VAR_COUNT, CV_32FC1,trainDatas);
	Mat trainLabelMat(TRAINSET_SIZE, 1, CV_32SC1,	trainLabels);

	double timeStart = static_cast<double>(cv::getTickCount());
	/*
	高斯核，参数效果：
	1. gamma=0.5, C=1;	错误率7.x%。
	2. gamma=0.5, C=20;	6.x%。
	3. gamma=0.5, C=80;	6.x%。
	4. gamma=0.5, C=160.
	5. gamma=1, C=20.	8.x%。
	6. gamma=0.25,C=20.	5.x%.
	*/
	Ptr<SVM> svm = SVM::create();
	svm->setKernel(SVM::RBF);
    svm->setType(SVM::C_SVC);	//支持向量机的类型。
    svm->setGamma(0.25);
    svm->setC(20);	//惩罚因子。
    svm->setTermCriteria(TermCriteria( CV_TERMCRIT_ITER, 100, FLT_EPSILON ));

	svm->train(trainMat, ROW_SAMPLE, trainLabelMat);
	double duration = (cv::getTickCount() - timeStart) / cv::getTickFrequency();
	cout<<"UCI letter recognition train elapse "<<duration<<" seconds."<<endl;
	char realLabel,predictLabel;
	float response;
	int errorCnt=0;
	for(int i=0; i!=TESTSET_SIZE; i++)	{
		realLabel = testLabels[i];
		Mat sample= (Mat_<float>(1,VAR_COUNT,testDatas[i]));
		response = svm->predict(sample);
		predictLabel = static_cast<char>(response);
		//printf("test label is %c, predict label is %c.\n", realLabel, predictLabel);

		if (realLabel!=static_cast<char>(response))	{
			printf("test label is %c, BUT predict label is %c.\n", realLabel, predictLabel);
			errorCnt++;
		}
	}
	float testErrorRate = float(errorCnt)/TESTSET_SIZE*100.0f;
	cout<<"UCI letter recognition SVM error rate:"<<testErrorRate<<"%, test size:"<<TESTSET_SIZE<<", errorCnt:"<<errorCnt<<endl;	//大约20%。

	return 0;
}

