#ifndef _OPENCV_IMAGESTORAGE_H_
#define _OPENCV_IMAGESTORAGE_H_

#include "highgui.h"



class CvCascadeImageReader
{
public:
    bool create( const std::string _posFilename, const std::string _negFilename, cv::Size _winSize );
    void restart() { posReader.restart(); }
    bool getNeg(cv::Mat &_img) { return negReader.get( _img ); }
    bool getPos(cv::Mat &_img) { return posReader.get( _img ); }

private:
    class PosReader
    {
    public:
        PosReader();
        virtual ~PosReader();
        bool create( const std::string _filename );
        bool get( cv::Mat &_img );
        void restart();

        short* vec;
        FILE*  file;	//注意指针file只对PosReader使用，PosReader只是使用了fstream对象的临时变量file
        int    count;
        int    vecSize;	////vecSize等于训练的命令参数(-w)*(-h)
        int    last;
        int    base;
    } posReader;

    class NegReader
    {
    public:
        NegReader();
        bool create( const std::string _filename, cv::Size _winSize );
        bool get( cv::Mat& _img );
        bool nextImg();

        cv::Mat     src, img;
        std::vector<std::string> imgFilenames;	//将neg.txt中收录的负样本名全部导入容器内
        cv::Point   offset, point;
        float   scale;
        float   scaleFactor;
        float   stepFactor;
        size_t  last, round;
        cv::Size    winSize;
    } negReader;
};

#endif
