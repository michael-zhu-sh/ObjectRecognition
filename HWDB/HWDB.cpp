// HWDB.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include <fstream>
#include <iostream>
#include <io.h>
#include <sstream>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

struct GNT_HEADER {
	unsigned int sampleSize;
	char tagCode[2];
	unsigned short width;
	unsigned short height;
};

const string GNT_TRAINSET_PATH = "C:/MLDataset/CASIA/HWDB1.1/train";
const int GNT_HEADER_SIZE = 10;
const int FEATURE_VECTOR_SIZE = 3240;	//一个汉字的特征向量维度。
uchar fileBuf[1024*30];	//汉字图像缓存，每个汉字大小不同。


/*
从1个GNT文件中解析汉字图像和编码。
images:存放解析出来的汉字图像。
*/
int readAGnt(const string &gntFilepath, vector<Mat> &retImages, vector<int> &retLabels)	{
	const Size dsize(70,77);	//汉字平均行height、列width是77、70。
	GNT_HEADER header;
	stringstream ss;
	string filename;
	unsigned short tag,ltag,rtag;
	int sampleCnt=0,imgSize=0,label;
	int rowSum=0,rowMean;	//计算汉字样本图像的行数均值(78)。
	int colSum=0,colMean;	//计算汉字样本图像的列数均值(67)。
	ifstream is(gntFilepath, ios::in | ios::binary );
	if(!is)
    {
        cerr << "FAIL to open input GNT file!" << endl;
        return 1;
    } else {
		cout<<"Succeed to open input GNT file."<<endl;
	}
	is.read((char*)&header, GNT_HEADER_SIZE);	//读取该GNT的第1个汉字。
	while (is)	{
		imgSize = header.width * header.height;
		//printf("%d chinese %c%c img size %d, rows %d, cols %d.\n", sampleCnt, header.tagCode[0], header.tagCode[1], imgSize, header.height, header.width);
		CV_Assert(imgSize<30720);
		is.read((char*)fileBuf, imgSize);
		if (!is)	{
			cerr << "error: only " << is.gcount() << " could be read"<<endl;
			break;
		} else {
			rowSum += header.height;
			colSum += header.width;
			sampleCnt++;
		}

		Mat originalImg = (Mat_<uchar>(header.height, header.width, fileBuf));	//GNT中的原始汉字图像，可能需要二值化。
		//如果resize到(77,63)，则一个汉字的HOG特征维度=(1+(77-14)/7)*(1+(77-14)/7)*(14*14/7/7*nbins)=(1+9)*(1+9)*(2*2*9)=10*10*36=3600D.
//如果resize到(77,70)，则一个汉字的HOG特征维度=(1+(77-14)/7)*(1+(70-14)/7)*(14*14/7/7*nbins)=(1+9)*(1+8)*(2*2*9)=10*9*36=3240D.
		Mat resizeImg;
		cv::resize(originalImg, resizeImg, dsize);
		retImages.push_back(resizeImg);
		
		tag = header.tagCode[0];
		ltag = tag << 8;
		tag = header.tagCode[1];
		rtag= tag;
		label = ltag | rtag;
		retLabels.push_back(label);
		/*
		Mat thresImg;
		cv::threshold(cImg, thresImg, 130, 255, THRESH_BINARY_INV);
		*/
		/*
		ss<<sampleCnt++;
		ss>>filename;
		filename = "output/trainsample" + filename + ".png";
		imwrite(filename,	resizeImg);
		if (sampleCnt>10)	break;
		ss.clear();
		*/

		is.read((char*)&header, GNT_HEADER_SIZE);	//继续读取GNT中的后续汉字。
	}

	is.close();

	rowMean = rowSum / sampleCnt;
	colMean = colSum / sampleCnt;
	cout<<"Succeed to read "<<sampleCnt<<" CHINESE samples from GNT "<<gntFilepath<<", row mean:"<<rowMean<<", col mean:"<<colMean<<endl;

	return 0;
}

/*
计算1个GNT中汉字的所有HOG特征。
*/
int computeHog(const vector<Mat> &images, Mat &trainFeatureMat)	{
	//根据images中的图像的大小77行x70列来设定。
	const int imgCnt = images.size();
	const Size winSize(70,77);
	const Size blockSize(14,14);
	const Size blockStride(7,7);
	const Size cellSize(7,7);
	const int nbins = 9;
	cv::HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, nbins);
	vector<float> descriptors;
	for(int i=0; i!=imgCnt; i++)	{
		hog.compute(images[i], descriptors, Size(1,1));
		Mat tmp = Mat(descriptors, true).t();
		tmp.row(0).copyTo(trainFeatureMat.row(i));	//保存HOG特征向量。

		descriptors.clear();
	}



	return 0;
}

/*
找到指定目录下所有的GNT文件。
*/
void getAllFiles(const string &path, vector<string> &files)
{
    //文件句柄  
    long   hFile = 0;
    //文件信息  
    struct _finddata_t fileinfo;  //很少用的文件信息读取结构
    string p;  //string类很有意思的一个赋值函数:assign()，有很多重载版本
    if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
    {
        do
        {
            if ((fileinfo.attrib &  _A_SUBDIR))  //比较文件类型是否是文件夹
            {
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
                {
                    files.push_back(p.assign(path).append("/").append(fileinfo.name));
                    getAllFiles(p.assign(path).append("/").append(fileinfo.name), files);
                }
            }
            else
            {
                files.push_back(p.assign(path).append("/").append(fileinfo.name));
            }
        } while (_findnext(hFile, &fileinfo) == 0);  //寻找下一个，成功返回0，否则-1
        _findclose(hFile);
    }
}

Mat trainFeatureMat(0, 3240, CV_32FC1);//存放训练集所有的HOG特征向量。
Mat trainLabelMat(0, 1, CV_32SC1);
int _tmain(int argc, _TCHAR* argv[])
{
	vector<string> gntTrainFiles;
	getAllFiles(GNT_TRAINSET_PATH, gntTrainFiles);
	const int trainGntSize = gntTrainFiles.size();

	double timeStart,duration;
	vector<Mat> images;
	vector<int> labels;
	int i,j,subTrainSize,totalTrainSize;	
	for(i=0; i!=trainGntSize; i++)	{
		timeStart = static_cast<double>(cv::getTickCount());
		cout<<i<<" begin to read chinese handwritten samples from file "<<gntTrainFiles[i]<<endl;
		readAGnt(gntTrainFiles[i], images, labels);
		subTrainSize = images.size();	//当前GNT包含的汉字数量。
		duration = (cv::getTickCount() - timeStart) / cv::getTickFrequency();
		cout<<"It takes "<<duration<<" seconds to read."<<endl;

		timeStart = static_cast<double>(cv::getTickCount());
		Mat featureMat(subTrainSize, 3240, CV_32FC1);
		computeHog(images, featureMat);
		CV_Assert(featureMat.rows==subTrainSize && featureMat.cols==FEATURE_VECTOR_SIZE);
		totalTrainSize	= trainFeatureMat.rows;
		trainFeatureMat.push_back(featureMat);
		CV_Assert(trainFeatureMat.rows==subTrainSize+totalTrainSize);

		Mat labelMat(subTrainSize, 1, CV_32SC1);
		for(j=0; j!=subTrainSize; j++)	{
			labelMat.ptr<int>(j)[0] = static_cast<int>(labels[j]);	//保存分类结果。
		}
		trainLabelMat.push_back(labelMat);
		duration = (cv::getTickCount() - timeStart) / cv::getTickFrequency();
		cout<<"It takes "<<duration<<" seconds to compute hog features and save them to trainFeatureMat+trainLabelMat."<<endl;

		featureMat.release();
		labelMat.release();
		labels.clear();
		images.clear();
	}
	cout<<"trainset feature mat rows:"<<trainFeatureMat.rows<<endl;
	/*
	vector<Mat> images;
	vector<int> labels;
	double timeStart = static_cast<double>(cv::getTickCount());
	readAGnt(GNT_TRAINSET_PATH, images, labels);
	const int trainSize = images.size();
	double duration = (cv::getTickCount() - timeStart) / cv::getTickFrequency();
	cout<<GNT_TRAINSET_PATH<<" include "<<images.size()<<" images, "<<labels.size()<<" labels, it take "<<duration<<" seconds."<<endl;

	timeStart = static_cast<double>(cv::getTickCount());
	Mat trainFeatureMat(trainSize, 3240, CV_32FC1, Scalar::all(0));//HOG特征向量矩阵。
    Mat trainLabelMat(trainSize, 1, CV_32SC1);
	computeHog(images, trainFeatureMat);
	for(int i=0; i!=trainSize; i++)	{
		trainLabelMat.ptr<int>(i)[0] = static_cast<int>(labels[i]);	//保存分类结果。
	}
	duration = (cv::getTickCount() - timeStart) / cv::getTickFrequency();
	cout<<"It take "<<duration<<" seconds to compute hog features."<<endl;
	*/

	return 0;
}

