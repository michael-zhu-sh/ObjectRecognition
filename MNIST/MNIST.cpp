/*
Author:Michael Zhu.
QQ:1265390626.
Email:1265390626@qq.com
QQ群:279441740
GIT:https://github.com/michael-zhu-sh/ObjectRecognition
本项目使用累计直方图特征和HOG特征这2种不同的图像特征，使用OpenCV提供的KNN和SVM这2种不同的分类器，
在MNIST手写体数字数据集上，进行了识别率的对比，完成了手写字符识别的第一个实验。
MNIST数据集官网http://yann.lecun.com/exdb/mnist/
*/
#include "stdafx.h"
#include <iostream>
#include <sstream>

#include <opencv2/opencv.hpp>
#include "opencv2/datasets/or_mnist.hpp"

#define VVP std::vector<std::vector<cv::Point>>

using namespace std;
using namespace cv;
using namespace cv::datasets;
using namespace cv::ml;


const string MNIST_PATH("../MNIST.dataset/");
const int H_FEATURE_VAR_COUNT = 28;	//水平方向投影特征变量数量，手写数字图像的大小是28x28。
const int V_FEATURE_VAR_COUNT = 28;	//垂直方向投影特征变量数量。
const bool SVM_TRAINED = false;
cv::Ptr<KNearest> knn;
cv::Ptr<SVM> linearSvmPtr;
cv::Ptr<SVM> rbfSvmPtr;


/*
根据训练集中的样本，计算水平或者垂直投影累计直方图。
img:原始手写数字图像。
features:存放返回的图像特征（28个）。
VorH:0垂直，1水平。
*/
int calcAccuHistFeature(const Mat &img, float *features, const int VorH=1)	{
	int size = img.rows;
	if (0!=VorH)	{
		size = img.rows;
	}

	Mat hist = Mat::zeros(1, size, CV_32FC1);
	float* rowPtr = hist.ptr<float>(0);
	float max = 0.0f,histV;
	int i;
	for(i=0; i!=size; i++)	{
		Mat data = (VorH==0)?img.col(i):img.row(i);
		histV = static_cast<float>(countNonZero(data));
		rowPtr[i] = histV;
		if (histV>max)	{
			max = histV;
		}
	}

	for(i=0; i!=size; i++)	{
		histV = rowPtr[i];
		histV = histV / max;
		features[i] = histV;
	}	//特征归一化。

	return 0;
}

/*
训练KNN模型。
float cv::ml::KNearest::findNearest 
(
InputArray samples, 
int k, 
OutputArray results, 
OutputArray neighborResponses = noArray(), 
OutputArray dist = noArray() 
)
*/
int knnTrain(Mat &trainData, Mat &trainLabel)	{
	const int K = 7;
	knn = KNearest::create();  
    knn->setDefaultK(K);
    knn->setIsClassifier(true);  
    knn->setAlgorithmType(KNearest::BRUTE_FORCE);  
	knn->train(trainData, cv::ml::ROW_SAMPLE, trainLabel);  

	return 0;
}


/*
预测charImg对应的分类。
*/
char knnPredict(const Mat &charImg, const int VorH=1)	{
	float features[H_FEATURE_VAR_COUNT];
	int ret = calcAccuHistFeature(charImg, features, VorH);
	char label;
	if (knn->isTrained())	{
		Mat sample(1, H_FEATURE_VAR_COUNT, CV_32FC1);
		for(int i=0; i!=H_FEATURE_VAR_COUNT; i++)	{
			sample.ptr<float>(0)[i] = features[i];
		}
		label = static_cast<char>(knn->predict(sample));
	} else {
		cerr<<"KNN is NOT trained, FAIL to predict."<<endl;
	}
	
	return label;
}

/*
批量预测。
result:保存预测结果。
float CvKNearest::find_nearest( const CvMat* _samples, int k, CvMat* results=0, const float** neighbors=0, CvMat* neighbor_responses=0, CvMat* dist=0 ) const;
参数说明：
samples为样本数*特征数的浮点矩阵；
K为寻找最近点的个数；results与预测结果；
neibhbors为k*样本数的指针数组（输入为const，实在不知为何如此设计）；
neighborResponse为样本数*k的每个样本K个近邻的输出值；
dist为样本数*k的每个样本K个近邻的距离。*/
int knnBatchPredict(Mat &testsetMat, Mat &resultMat)	{
	knn->predict(testsetMat, resultMat);
	return 0;
}

/*
分析特征矩阵。
featureMat:1行为1个样本，每列是1个特征变量。
*/
int analyzeFeatureMat(Mat &featureMat)	{
	const int rows = featureMat.rows;
	cv::Scalar     mean;  
    cv::Scalar     stddev;  
	float m,s;
	for(int i=0; i!=featureMat.cols; i++)	{
		Mat tmp = Mat(rows, 1, CV_32FC1);
		featureMat.col(i).copyTo(tmp);
		cv::meanStdDev(tmp, mean, stddev);
		m = static_cast<float>(mean.val[0]);
		s = static_cast<float>(stddev.val[0]);
		cout<<"feature matrix column ["<<i<<"] mean:"<<m<<", stddev:"<<s<<"."<<endl;
	}

	return 0;
}

/*
使用KNN分类器，训练、预测MNIST数据集。使用了单方向（水平）的特征。
*/
int knnMnist()	{
	Ptr<OR_mnist> mnistPtr = OR_mnist::create();
	mnistPtr->load(MNIST_PATH);
	vector< Ptr<Object> > trainSet	= mnistPtr->getTrain();
	vector< Ptr<Object> > testSet	= mnistPtr->getTest();
	const int trainSize = trainSet.size();
	const int testSize	= testSet.size();
	cout<<"trainset size:"<<trainSize<<", testset size:"<<testSize<<", begin to train by KNN."<<endl;
	//以上从4个文件中加载MNIST数据集。
	stringstream ss;
	string filename;
	int i,j,pos;
	int errorCnt=0;	//预测错误的数量。
	OR_mnistObj *sample;
	float features[H_FEATURE_VAR_COUNT];	//存放每个数字图像的特征。
	char label;	//存放数字图像对应的分类结果。
	float *trainDataRowPtr;
	Mat trainDataMat(trainSize, H_FEATURE_VAR_COUNT, CV_32FC1), trainLabelMat(trainSize, 1, CV_32FC1);
	double timeStart = static_cast<double>(cv::getTickCount());
	for(i=0; i!=trainSize; i++)	{
		sample	= static_cast<OR_mnistObj *>(trainSet[i].get());
		label	= sample->label;
		/*
		if (i<100)	{
			ss<<i;
			ss>>filename;
			filename = "c:/ImageDatabase/MNIST/images/train" + filename + ".png"; 
			imwrite(filename, sample->image);
			ss.clear();
		}
		*/
		calcAccuHistFeature(sample->image, features);	//水平方向的直方图识别正确率高。
		trainDataRowPtr = trainDataMat.ptr<float>(i);
		for(j=0; j!=H_FEATURE_VAR_COUNT; j++)	{
			trainDataRowPtr[j] = features[j];
		}
		//以上保存特征数据。
		trainLabelMat.ptr<float>(i)[0] = static_cast<float>(label);	//保存分类结果。
	}
	//analyzeFeatureMat(trainDataMat);	//分析特征矩阵。
	knnTrain(trainDataMat, trainLabelMat);
	double duration = (cv::getTickCount() - timeStart) / cv::getTickFrequency();
	cout<<"MNIST KNN train elapse:"<<duration<<" seconds."<<endl;	//整个训练集上的训练耗时，大概2秒多。

	cout<<"begin to test "<<testSize<<" samples."<<endl;
	timeStart = static_cast<double>(cv::getTickCount());
	Mat testSamples(testSize, H_FEATURE_VAR_COUNT, CV_32FC1);
	Mat testResults(testSize, 1, CV_32FC1);
	float *testDataRowPtr;
	char *cRealResults	= new char[testSize];
	for(i=0; i!=testSize; i++)	{
		sample	= static_cast<OR_mnistObj *>(testSet[i].get());
		testDataRowPtr	= testSamples.ptr<float>(i);
		cRealResults[i]	= sample->label;	//实际的类别。
		calcAccuHistFeature(sample->image, features);	//水平方向的直方图识别正确率高。
		for(j=0; j!=H_FEATURE_VAR_COUNT; j++)	{
			testDataRowPtr[j] = features[j];
		}
	}
	knnBatchPredict(testSamples, testResults);	//批量预测。
	duration = (cv::getTickCount() - timeStart) / cv::getTickFrequency();
	cout<<"MNIST KNN predict elapse:"<<duration<<" seconds."<<endl;	//整个测试集上的预测耗时，大概20多秒。
	
	char cReal,cTest;
	vector<char> errorChars;
	vector<char>::iterator it;
	vector<int> errorNums;	//样本被错误分类的累计次数。
	for(i=0; i!=testSize; i++)	{
		cReal = cRealResults[i];
		cTest = static_cast<char>(testResults.ptr<float>(i)[0]);
		if (cReal != cTest )	{
			it = std::find(errorChars.begin(), errorChars.end(), cReal); 
			if (it!=errorChars.end())	{
				pos = std::distance(errorChars.begin(), it);
				errorNums[pos] = errorNums[pos] + 1; 
			} else {
				errorChars.push_back(cReal);
				errorNums.push_back(1);
			}

			errorCnt++;
		}
	}
	//以上是统计分类错误率。
	float testErrorRate = float(errorCnt)/testSize*100.0f;
	cout<<"KNN error rate:"<<testErrorRate<<"%."<<endl;	//19.x%。
	for(i=0; i!=errorChars.size(); i++)	{
		printf("char [%u] wrongly classify [%d].\n", errorChars[i], errorNums[i]);
	}

	cout<<"Finish to train and predict MNIST dataset by KNN."<<endl;

	return 0;
}


/*
SVM分类器的使用DEMO。
*/
int svmDemo()	{
    int width = 512, height = 512;
    Mat image = Mat::zeros(height, width, CV_8UC3);

    int labels[4] = {1, -1, -1, -1};	//1 green, -1 blue.
    float trainingData[4][2] = { {501, 10}, {255, 10}, {501, 255}, {10, 501} };
    Mat trainingDataMat(4, 2, CV_32FC1, trainingData);
    Mat labelsMat(4, 1, CV_32SC1, labels);
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);
    Vec3b green(0,255,0), blue (255,0,0);
    for (int i = 0; i < image.rows; ++i)
        for (int j = 0; j < image.cols; ++j)
        {
            Mat sampleMat = (Mat_<float>(1,2) << j,i);
			//out<<sampleMat<<std::endl;
            float response = svm->predict(sampleMat);	//预测sampleMat的分类标签label。
            if (response == 1)
                image.at<Vec3b>(i,j)  = green;
            else if (response == -1)
                image.at<Vec3b>(i,j)  = blue;
        }
    int thickness = -1;
    int lineType = 8;
    circle(	image, Point(501,  10), 5, Scalar(  0,   0,   0), thickness, lineType );
    circle(	image, Point(255,  10), 5, Scalar(255, 255, 255), thickness, lineType );
    circle(	image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType );
    circle(	image, Point( 10, 501), 5, Scalar(255, 255, 255), thickness, lineType );
    thickness = 2;
    lineType  = 8;
    Mat sv = svm->getUncompressedSupportVectors();

    for (int i = 0; i < sv.rows; ++i)
    {
        const float* v = sv.ptr<float>(i);
        circle(	image,  Point( (int) v[0], (int) v[1]),   6,  Scalar(128, 128, 128), thickness, lineType);
    }
    imshow("SVM Simple Example", image); // show it to the user
    waitKey(0);

	return 0;
}

/*
使用SVM分类器测试MNIST数据集。
核函数的选择：
KernelTypes { 
  CUSTOM =-1, 
  LINEAR =0,	线性核
  POLY =1,	多项式核
  RBF =2,	高斯核（径向基函数核）
  SIGMOID =3, 
  CHI2 =4, 
  INTER =5 
}
支持向量机的选择：
C_SVC,	C-Support Vector Classification. n-class classification (n ≥ 2), allows imperfect separation of classes with penalty multiplier C for outliers. 
	C类支撑向量分类机。 n类分组 （n≥2），容许用异常值处罚因子C进行不完全分类。
NU_SVC,	ν-Support Vector Classification. n-class classification with possible imperfect separation. Parameter ν(in the range 0..1, the larger the value, the smoother the decision boundary) is used instead of C. 
	ν类支撑向量分类机。n类似然不完全分类的分类器。参数为庖代C（其值在区间【0，1】中，nu越大，决定计划鸿沟越腻滑）。
ONE_CLASS,
	单分类器，所有的练习数据提取自同一个类里，然后SVM建树了一个分界线以分别该类在特点空间中所占区域和其它类在特点空间中所占区域。
EPS_SVR,
	ϵ类支撑向量回归机。练习集中的特点向量和拟合出来的超平面的间隔须要小于p。异常值处罚因子C被采取。
NU_SVR,
	ν类支撑向量回归机。ν庖代了p。
*/
int svmMnist()	{
	Ptr<OR_mnist> mnistPtr = OR_mnist::create();
	mnistPtr->load(MNIST_PATH);
	vector< Ptr<Object> > trainSet	= mnistPtr->getTrain();
	vector< Ptr<Object> > testSet	= mnistPtr->getTest();
	const int trainSize = trainSet.size();
	const int testSize	= testSet.size();
	cout<<"trainset size:"<<trainSize<<", testset size:"<<testSize<<", begin to train by SVM."<<endl;
	//以上从4个文件中加载MNIST数据集。

	double timeStart = static_cast<double>(cv::getTickCount());
	int i,j;
	OR_mnistObj *sample;
	char label;
	float features[H_FEATURE_VAR_COUNT];
	float *trainDataRowPtr;
	const int trainsetRows	= trainSize;
	const int testsetRows	= testSize;
	const int trainsetCols	= H_FEATURE_VAR_COUNT;
	const int testsetCols	= trainsetCols;
	Mat trainDataMat(trainsetRows, trainsetCols, CV_32FC1);
    Mat trainLabelMat(trainsetRows, 1, CV_32SC1);
	if (!SVM_TRAINED)	{
	for(i=0; i!=trainSize; i++)	{
		sample	= static_cast<OR_mnistObj *>(trainSet[i].get());
		label	= sample->label;
	    //printf("sample %d, label: %u\n", i, sample->label);
		calcAccuHistFeature(sample->image, features);	//水平方向的直方图识别正确率高。
		trainDataRowPtr = trainDataMat.ptr<float>(i);
		for(j=0; j!=H_FEATURE_VAR_COUNT; j++)	{
			trainDataRowPtr[j] = features[j];
		}
		//以上保存特征数据。
		trainLabelMat.ptr<int>(i)[0] = static_cast<int>(label);	//保存分类结果。
	}
	}

	Ptr<SVM> svm = SVM::create();
	/*
	高斯核，参数效果：
	1. gamma=0.5, C=1;	错误率49%。
	2. gamma=0.5, C=20;	64%。
	3. gamma=0.5, C=80;	64%。
	4. gamma=0.5, C=160.
	5. gamma=5,	C=20.	43%。
	6. gamma=10,C=20.	58%.
	*/
	svm->setKernel(SVM::RBF);
    svm->setType(SVM::C_SVC);	//支持向量机的类型。
    svm->setGamma(10);
    svm->setC(20);	//惩罚因子。
    svm->setTermCriteria(TermCriteria( CV_TERMCRIT_ITER, 100, FLT_EPSILON ));

	if (!SVM_TRAINED)	{
		svm->train(trainDataMat, ROW_SAMPLE, trainLabelMat);
		svm->save("mnist.yml");
	} else {
		svm->load("mnist.yml");
	}
	double duration = (cv::getTickCount() - timeStart) / cv::getTickFrequency();
	cout<<"MNIST SVM train elapse:"<<duration<<" seconds."<<endl;	//整个训练集上的耗时，大概10秒。
	cout<<"Succeed to train by SVM."<<endl;

	timeStart = static_cast<double>(cv::getTickCount());
	char cReal,cTest;
	float response;
	int errorCnt=0;
	for(i=0; i!=testSize; i++)	{
		sample	= static_cast<OR_mnistObj *>(testSet[i].get());
		cReal	= sample->label;	//实际的类别。
		calcAccuHistFeature(sample->image, features);	//水平方向的直方图识别正确率高。
        //Mat sampleMat(1,testsetCols,CV_32F,features);
		Mat sampleMat = (Mat_<float>(1,28,features));
		response= svm->predict(sampleMat);	//预测sampleMat的分类标签label。
		cTest	= static_cast<char>(response);
		if (cTest!=cReal)	{
			//printf("real:%u, but predict:%u\n", cReal, cTest);
			errorCnt++;
		}
	}
	//以上是统计分类错误率。
	duration = (cv::getTickCount() - timeStart) / cv::getTickFrequency();
	float testErrorRate = float(errorCnt)/testSize*100.0f;
	cout<<"SVM error rate:"<<testErrorRate<<"%."<<endl;	//大约20%。
	cout<<"Elapse "<<duration<<" seconds to predict MNIST dataset by SVM."<<endl;

	return 0;
}

/*HOG+SVM---------------------------------------------------------------------------*/
/*
初始化2个SVM。
*/
int initSVM()	{
	/*
	高斯核，参数效果：
	1. gamma=0.5, C=1;	错误率49%。
	2. gamma=0.5, C=20;	64%。
	3. gamma=0.5, C=80;	64%。
	4. gamma=0.5, C=160.
	5. gamma=5,	C=20.	43%。
	6. gamma=10,C=20.	58%.
	*/
	rbfSvmPtr = SVM::create();
	rbfSvmPtr->setKernel(SVM::RBF);
    rbfSvmPtr->setType(SVM::C_SVC);	//支持向量机的类型。
    rbfSvmPtr->setGamma(5);
    rbfSvmPtr->setC(20);	//惩罚因子。
    rbfSvmPtr->setTermCriteria(TermCriteria( CV_TERMCRIT_ITER, 100, FLT_EPSILON ));

	linearSvmPtr = SVM::create();
	linearSvmPtr->setKernel(SVM::LINEAR);
    linearSvmPtr->setType(SVM::C_SVC);	//支持向量机的类型。
	linearSvmPtr->setC(20);
    linearSvmPtr->setTermCriteria(TermCriteria( CV_TERMCRIT_ITER, 100, FLT_EPSILON ));

	cout<<"Succeed to initialize 2 SVMs."<<endl;

	return 0;
}

/*
应用手写数字的HOG特征，使用SVM进行分类预测。
*/
int hogMnist()	{
	const int HOG_FEATURE_VECTOR_LEN = 3*3*36;
	Ptr<OR_mnist> mnistPtr = OR_mnist::create();
	mnistPtr->load(MNIST_PATH);
	vector< Ptr<Object> > trainSet	= mnistPtr->getTrain();
	vector< Ptr<Object> > testSet	= mnistPtr->getTest();
	const int trainSize = trainSet.size();
	const int testSize	= testSet.size();
	cout<<"trainset size:"<<trainSize<<", testset size:"<<testSize<<", begin to train by SVM+HOG."<<endl;

	//以上从4个文件中加载MNIST数据集。
	int i;
	OR_mnistObj *samplePtr;
	
	double timeStart = static_cast<double>(cv::getTickCount());
	//根据原始数字图像的大小28x28来设定。
	const Size winSize(28,28);
	const Size blockSize(14,14);
	const Size blockStride(7,7);
	const Size cellSize(7,7);
	const int nbins = 9;
	const int FEATURE_VECTOR_SIZE = (1+(28-14)/7)*(1+(28-14)/7)*(14*14/7/7*nbins);
	cv::HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, nbins);
	vector<float> descriptors;
	char label;
	Mat trainFeatureMat(trainSize, FEATURE_VECTOR_SIZE, CV_32FC1, Scalar::all(0));//HOG特征向量矩阵。
    Mat trainLabelMat(trainSize, 1, CV_32SC1);
	for(i=0; i!=trainSize; i++)	{
		samplePtr	= static_cast<OR_mnistObj *>(trainSet[i].get());
		label	= samplePtr->label;
		hog.compute(samplePtr->image, descriptors, Size(1,1));
		//cout<<"descriptors size:"<<descriptors.size()<<endl;

		Mat tmp = Mat(descriptors, true).t();
		tmp.row(0).copyTo(trainFeatureMat.row(i));	//保存HOG特征向量。
		trainLabelMat.ptr<int>(i)[0] = static_cast<int>(label);	//保存分类结果。

		descriptors.clear();
	}
	double duration = (cv::getTickCount() - timeStart) / cv::getTickFrequency();
	cout<<"It takes "<<duration<<" seconds to generate feature matrics by HOG."<<endl;	//15.x seconds.

	initSVM();	//初始化SVMs分类器。

	timeStart = static_cast<double>(cv::getTickCount());
	linearSvmPtr->train(trainFeatureMat, ROW_SAMPLE, trainLabelMat);
	duration = (cv::getTickCount() - timeStart) / cv::getTickFrequency();
	cout<<"MNIST SVM(LINEAR)+HOG train elapse:"<<duration<<" seconds."<<endl;	//33.x seconds.
	/*
	timeStart = static_cast<double>(cv::getTickCount());
	rbfSvmPtr->train(trainFeatureMat, ROW_SAMPLE, trainLabelMat);
	duration = (cv::getTickCount() - timeStart) / cv::getTickFrequency();
	cout<<"MNIST SVM(RBF)+HOG train elapse:"<<duration<<" seconds."<<endl;
	*/
	timeStart = static_cast<double>(cv::getTickCount());
	char cReal,cTest;
	float response;
	int pos,errorCnt=0;	//预测错误的数量。
	vector<char> errorChars;
	vector<char>::iterator it;
	vector<int> errorNums;	//样本被错误分类的累计次数。
	for(i=0; i!=testSize; i++)	{
		samplePtr	= static_cast<OR_mnistObj *>(testSet[i].get());
		cReal	= samplePtr->label;	//实际的类别。
		hog.compute(samplePtr->image, descriptors, Size(1,1));
		
		Mat tmp = Mat(descriptors, true).t();
		response= linearSvmPtr->predict(tmp);	//预测sampleMat的分类标签label。
		cTest	= static_cast<char>(response);
		if (cTest!=cReal)	{
			it = std::find(errorChars.begin(), errorChars.end(), cReal); 
			if (it!=errorChars.end())	{
				pos = std::distance(errorChars.begin(), it);
				errorNums[pos] = errorNums[pos] + 1; 
			} else {
				errorChars.push_back(cReal);
				errorNums.push_back(1);
			}

			errorCnt++;
		}
	}
	//以上是统计分类错误率。
	duration = (cv::getTickCount()-timeStart) / cv::getTickFrequency();
	float testErrorRate = float(errorCnt)/testSize*100.0f;
	cout<<"SVM(linear)+HOG error rate:"<<testErrorRate<<"%."<<endl;	//3.x%。
	for(i=0; i!=errorChars.size(); i++)	{
		printf("char [%u] wrongly classify [%d].\n", errorChars[i], errorNums[i]);
	}
	cout<<"Elapse "<<duration<<" seconds to predict MNIST dataset by SVM(linear)+HOG."<<endl;	//2.x seconds.
	/*
	errorCnt=0;	//预测错误的数量。
	for(i=0; i!=testSize; i++)	{
		samplePtr	= static_cast<OR_mnistObj *>(testSet[i].get());
		cReal	= samplePtr->label;	//实际的类别。
		hog.compute(samplePtr->image, descriptors, Size(1,1));
		
		Mat tmp = Mat(descriptors, true).t();
		response= rbfSvmPtr->predict(tmp);	//预测sampleMat的分类标签label。
		cTest	= static_cast<char>(response);
		if (cTest!=cReal)	{
			errorCnt++;
		}
	}
	//以上是统计分类错误率。
	duration = (cv::getTickCount()-timeStart) / cv::getTickFrequency();
	testErrorRate = float(errorCnt)/testSize*100.0f;
	cout<<"SVM(rbf)+HOG error rate:"<<testErrorRate<<"%."<<endl;	//23.x%。
	cout<<"Elapse "<<duration<<" seconds to predict MNIST dataset by SVM(rbf)+HOG."<<endl;	//45.x seconds.
	*/

  	return 0;
}

int _tmain(int argc, _TCHAR* argv[])
{
	hogMnist();

	//knnMnist();

	//svmMnist();

	return 0;
}
