#include<ostream>
#include <iostream>
#include <stdlib.h>
#include<vector>
#include <string>
#include<fstream>
#include<algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;
#define Random(low, up) (rand()%(up-low+1)) + low - 1

vector<cv::Point2f> getRotatePoint(int row, vector<cv::Point2f> Points, const cv::Point rotate_center, const double angle) {
    vector<cv::Point2f> dstPoints;
    float x1 = 0.f, y1 = 0.f;
    for (size_t i = 0; i < Points.size(); i++){
        x1 = Points.at(i).x;
        y1 = row - Points.at(i).y;
        float x2 = rotate_center.x;
        float y2 = row - rotate_center.y;
        float x = (x1 - x2)*cos(M_PI / 180.0 * angle) - (y1 - y2)*sin(M_PI / 180.0 * angle) + x2;
        float y = (x1 - x2)*sin(M_PI / 180.0 * angle) + (y1 - y2)*cos(M_PI / 180.0 * angle) + y2;
        y = row - y;
        dstPoints.push_back(Point2i(x, y));
    }
    return dstPoints;
}
Mat rotateImg(Mat src, int angle){
    Mat dst;
    //填充图像
    int maxBorder =(int) (max(src.cols, src.rows)* 1.414 ); //即为sqrt(2)*max
    int dx = (maxBorder - src.cols)/2;
    int dy = (maxBorder - src.rows)/2;
    copyMakeBorder(src, dst, dy, dy, dx, dx, BORDER_CONSTANT);
    //旋转
    Point2f center( (float)(dst.cols/2) , (float) (dst.rows/2));
    Mat affine_matrix = getRotationMatrix2D( center, angle, 1.0 );//求得旋转矩阵
    warpAffine(dst, dst, affine_matrix, dst.size());
    return dst;
}

Mat rotateSameImg(Mat src, int angle){
    Mat dst;
    //旋转
    Point2f center( (float)(src.cols/2) , (float) (src.rows/2));
    Mat affine_matrix = getRotationMatrix2D( center, angle, 1.0 );//求得旋转矩阵
    warpAffine(src, dst, affine_matrix, dst.size());
    return dst;
}

vector<cv::Point2f> CropImg(Mat src, Mat& dst , int angle, vector<cv::Point2f> Points){
    float radian = (float) (angle /180.0 * M_PI);
    vector<cv::Point2f> dstPoints;
    //计算图像旋转之后包含图像的最大的矩形
    float sinVal = fabs(sin(radian));
    float cosVal = fabs(cos(radian));
    Size targetSize( (int)(src.cols * cosVal +src.rows * sinVal),
                     (int)(src.cols * sinVal + src.rows * cosVal) );
    //剪掉多余边框

    int x = (dst.cols - targetSize.width) / 2;
    int y = (dst.rows - targetSize.height) / 2;
    Rect rect(x, y, targetSize.width, targetSize.height);
    dst = Mat(dst,rect);
    for(int i=0; i<Points.size(); i++){
        float point_x =  Points.at(i).x - x;
        float point_y = Points.at(i).y - y;
        dstPoints.push_back(Point2f(point_x, point_y));
    }
    return dstPoints;
}

void RotateBatchImage(string& srcfilePath, string &labelPath, string &newTrainingSetFile){
    ofstream setFile(newTrainingSetFile, ios::out|ios::app);
    if(!setFile.is_open()){
        std::cout<< "cannot open lablefile , the file" << newTrainingSetFile << "does not exit!"<<std::endl;
        return;
    }
    Mat src = imread(srcfilePath);
    vector<cv::Point2f> Points;
    int maxBorder =(int) (max(src.cols, src.rows)* 1.414 );
    int dx = (maxBorder - src.cols)/2;
    int dy = (maxBorder - src.rows)/2;
    std::ifstream infile(labelPath.c_str());
    std::string lineStr ;
    std::stringstream sstr ;
    if (!infile.good()) {
        std::cout << "Cannot open " << labelPath<<std::endl;
        return;
    }
    float x[5];
    float y[5];
    int gender;
    int glass;
    while (std::getline(infile, lineStr )) {
        sstr << lineStr;
        sstr >> x[0] >> x[1] >> x[2] >> x[3] >> x[4] >> y[0] >> y[1] >> y[2] >> y[3] >> y[4] >> gender >> glass;
        for(size_t ii=0; ii<5; ii++){
            x[ii] += dx;
            y[ii] += dy;
            Points.push_back(cv::Point2f(x[ii], y[ii]));
        }
        lineStr.clear();
    }
    double angle ;
    srand((int)time(0));
    size_t iPos = srcfilePath.find(".jpg");
    string s2 = srcfilePath.substr(0, iPos);
    for(int iter = 0; iter < 15; iter++){
        string newImgFilepath = s2 + "_"+std::to_string(iter)+".jpg";
        string newImgFilepath_rotete = s2 + "_"+std::to_string(iter)+"_rotate.jpg";
        string newLablePath = labelPath + "_"+std::to_string(iter);
        angle = Random(-120, 120);
        std::cout<<"angle: "<<angle<<std::endl;
        Mat dst =  rotateImg(src, angle);
        cv::Point2f center(dst.cols / 2., dst.rows / 2.);
        vector<cv::Point2f> dstPoints = getRotatePoint(dst.rows, Points, center, angle);
        vector<cv::Point2f> cropPoints = CropImg(src, dst , angle, dstPoints);
        ofstream file(newLablePath, ios::out);
        if(!file.is_open()){
            std::cout<< "cannot open lablefile , the file" << newLablePath << "does not exit!"<<std::endl;
            return;
        }
        file << cropPoints.at(0).x << " " << cropPoints.at(1).x << " " << cropPoints.at(2).x << " " << cropPoints.at(3).x << " "
            << cropPoints.at(4).x << " " << cropPoints.at(0).y << " " << cropPoints.at(1).y << " " << cropPoints.at(2).y << " " << cropPoints.at(3).y << " "
            << cropPoints.at(4).y << " "<<gender<<" "<<glass<< std::endl;
        cv::imwrite(newImgFilepath, dst);
        for(size_t ii=0; ii<cropPoints.size(); ii++){
            cv::circle(dst, cropPoints.at(ii), 5, cv::Scalar(0, 0, 255), 2);
        }
        cv::imwrite(newImgFilepath_rotete, dst);
        file.close();
        setFile << newImgFilepath <<std::endl;
    }
    setFile.close();
}

void RotateSameBatchImage(string& srcfilePath, string &labelPath, string &newTrainingSetFile){
    ofstream setFile(newTrainingSetFile, ios::out|ios::app);
    if(!setFile.is_open()){
        std::cout<< "cannot open lablefile , the file" << newTrainingSetFile << "does not exit!"<<std::endl;
        return;
    }
    Mat src = imread(srcfilePath);
    vector<cv::Point2f> Points;
    std::ifstream infile(labelPath.c_str());
    std::string lineStr ;
    std::stringstream sstr ;
    if (!infile.good()) {
        std::cout << "Cannot open " << labelPath<<std::endl;
        return;
    }
    float x[5];
    float y[5];
    int gender;
    int glass;
    while (std::getline(infile, lineStr )) {
        sstr << lineStr;
        sstr >> x[0] >> x[1] >> x[2] >> x[3] >> x[4] >> y[0] >> y[1] >> y[2] >> y[3] >> y[4] >> gender >> glass;
        for(size_t ii=0; ii<5; ii++){
            Points.push_back(cv::Point2f(x[ii], y[ii]));
        }
        lineStr.clear();
    }
    double angle ;
    srand((int)time(0));
    size_t iPos = srcfilePath.find(".jpg");
    string s2 = srcfilePath.substr(0, iPos);
    for(int iter = 0; iter < 5; iter++){
        string newImgFilepath = s2 + "_"+std::to_string(iter)+".jpg";
        string newImgFilepath_rotete = s2 + "_"+std::to_string(iter)+"_rotate.jpg";
        string newLablePath = labelPath + "_"+std::to_string(iter);
        angle = Random(-120, 120);
        std::cout<<"angle: "<<angle<<std::endl;
        Mat dst =  rotateSameImg(src, angle);
        cv::Point2f center(dst.cols / 2., dst.rows / 2.);
        vector<cv::Point2f> dstPoints = getRotatePoint(dst.rows, Points, center, angle);
        ofstream file(newLablePath, ios::out);
        if(!file.is_open()){
            std::cout<< "cannot open lablefile , the file" << newLablePath << "does not exit!"<<std::endl;
            return;
        }
        file << dstPoints.at(0).x << " " << dstPoints.at(1).x << " " << dstPoints.at(2).x << " " << dstPoints.at(3).x << " "
            << dstPoints.at(4).x << " " << dstPoints.at(0).y << " " << dstPoints.at(1).y << " " << dstPoints.at(2).y << " " << dstPoints.at(3).y << " "
            << dstPoints.at(4).y << " "<<gender<<" "<<glass<< std::endl;
        cv::imwrite(newImgFilepath, dst);
        for(size_t ii=0; ii<dstPoints.size(); ii++){
            cv::circle(dst, dstPoints.at(ii), 5, cv::Scalar(0, 0, 255), 2);
        }
        cv::imwrite(newImgFilepath_rotete, dst);
        file.close();
        setFile << newImgFilepath <<std::endl;
    }
    setFile.close();
}


int main() {
    string srcTrainsetFile = "/home/stive/workspace/dataset/facedata/mtfl/ImageSets/training.txt";
    string srcTestsetFile = "/home/stive/workspace/dataset/facedata/mtfl/ImageSets/testing.txt";
    string newTrainsetFile = "/home/stive/workspace/dataset/facedata/mtfl/ImageSets/newtraining.txt";
    string newTestsetFile = "/home/stive/workspace/dataset/facedata/mtfl/ImageSets/newtesting.txt";
    std::ifstream infile(srcTrainsetFile.c_str());
    std::string lineStr ;
    if (!infile.good()) {
        std::cout << "Cannot open " << srcTrainsetFile;
        return 0;
    }
    string imgPath;
    string labelPath;
    while (!infile.eof()) {
        std::getline(infile, lineStr );
        std::stringstream sstr ;
        sstr << lineStr;
        sstr >> imgPath >> labelPath;
        std:: cout << "imgPath: "<<imgPath<<" labelPath: "<<labelPath<<std::endl;
        RotateBatchImage(imgPath, labelPath, newTrainsetFile);
        lineStr.clear();
        while(std::getline(infile, lineStr )){
            std::stringstream newsster(lineStr) ;
            newsster >> imgPath >> labelPath;
            std:: cout << "imgPath: "<<imgPath<<" labelPath: "<<labelPath<<std::endl;
            //RotateBatchImage(imgPath, labelPath, newTrainsetFile);
            RotateSameBatchImage(imgPath, labelPath, newTrainsetFile);
        }
    }
    infile.close();
    return 0;
}
