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
const int FEATURE_VECTOR_SIZE = 3240;	//һ�����ֵ���������ά�ȡ�
uchar fileBuf[1024*30];	//����ͼ�񻺴棬ÿ�����ִ�С��ͬ��


/*
��1��GNT�ļ��н�������ͼ��ͱ��롣
images:��Ž��������ĺ���ͼ��
*/
int readAGnt(const string &gntFilepath, vector<Mat> &retImages, vector<int> &retLabels)	{
	const Size dsize(70,77);	//����ƽ����height����width��77��70��
	GNT_HEADER header;
	stringstream ss;
	string filename;
	unsigned short tag,ltag,rtag;
	int sampleCnt=0,imgSize=0,label;
	int rowSum=0,rowMean;	//���㺺������ͼ���������ֵ(78)��
	int colSum=0,colMean;	//���㺺������ͼ���������ֵ(67)��
	ifstream is(gntFilepath, ios::in | ios::binary );
	if(!is)
    {
        cerr << "FAIL to open input GNT file!" << endl;
        return 1;
    } else {
		cout<<"Succeed to open input GNT file."<<endl;
	}
	is.read((char*)&header, GNT_HEADER_SIZE);	//��ȡ��GNT�ĵ�1�����֡�
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

		Mat originalImg = (Mat_<uchar>(header.height, header.width, fileBuf));	//GNT�е�ԭʼ����ͼ�񣬿�����Ҫ��ֵ����
		//���resize��(77,63)����һ�����ֵ�HOG����ά��=(1+(77-14)/7)*(1+(77-14)/7)*(14*14/7/7*nbins)=(1+9)*(1+9)*(2*2*9)=10*10*36=3600D.
//���resize��(77,70)����һ�����ֵ�HOG����ά��=(1+(77-14)/7)*(1+(70-14)/7)*(14*14/7/7*nbins)=(1+9)*(1+8)*(2*2*9)=10*9*36=3240D.
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

		is.read((char*)&header, GNT_HEADER_SIZE);	//������ȡGNT�еĺ������֡�
	}

	is.close();

	rowMean = rowSum / sampleCnt;
	colMean = colSum / sampleCnt;
	cout<<"Succeed to read "<<sampleCnt<<" CHINESE samples from GNT "<<gntFilepath<<", row mean:"<<rowMean<<", col mean:"<<colMean<<endl;

	return 0;
}

/*
����1��GNT�к��ֵ�����HOG������
*/
int computeHog(const vector<Mat> &images, Mat &trainFeatureMat)	{
	//����images�е�ͼ��Ĵ�С77��x70�����趨��
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
		tmp.row(0).copyTo(trainFeatureMat.row(i));	//����HOG����������

		descriptors.clear();
	}



	return 0;
}

/*
�ҵ�ָ��Ŀ¼�����е�GNT�ļ���
*/
void getAllFiles(const string &path, vector<string> &files)
{
    //�ļ����  
    long   hFile = 0;
    //�ļ���Ϣ  
    struct _finddata_t fileinfo;  //�����õ��ļ���Ϣ��ȡ�ṹ
    string p;  //string�������˼��һ����ֵ����:assign()���кܶ����ذ汾
    if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
    {
        do
        {
            if ((fileinfo.attrib &  _A_SUBDIR))  //�Ƚ��ļ������Ƿ����ļ���
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
        } while (_findnext(hFile, &fileinfo) == 0);  //Ѱ����һ�����ɹ�����0������-1
        _findclose(hFile);
    }
}

Mat trainFeatureMat(0, 3240, CV_32FC1);//���ѵ�������е�HOG����������
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
		subTrainSize = images.size();	//��ǰGNT�����ĺ���������
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
			labelMat.ptr<int>(j)[0] = static_cast<int>(labels[j]);	//�����������
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
	Mat trainFeatureMat(trainSize, 3240, CV_32FC1, Scalar::all(0));//HOG������������
    Mat trainLabelMat(trainSize, 1, CV_32SC1);
	computeHog(images, trainFeatureMat);
	for(int i=0; i!=trainSize; i++)	{
		trainLabelMat.ptr<int>(i)[0] = static_cast<int>(labels[i]);	//�����������
	}
	duration = (cv::getTickCount() - timeStart) / cv::getTickFrequency();
	cout<<"It take "<<duration<<" seconds to compute hog features."<<endl;
	*/

	return 0;
}

