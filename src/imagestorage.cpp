#include "opencv2/core/core.hpp"
#include "opencv2/core/internal.hpp"

#include "cv.h"
#include "imagestorage.h"
#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

bool CvCascadeImageReader::create( const string _posFilename, const string _negFilename, Size _winSize )
{
    return posReader.create(_posFilename) && negReader.create(_negFilename, _winSize);
}

CvCascadeImageReader::NegReader::NegReader()
{
    src.create( 0, 0 , CV_8UC1 );
    img.create( 0, 0, CV_8UC1 );
    point = offset = Point( 0, 0 );
    scale       = 1.0F;
    scaleFactor = 1.4142135623730950488016887242097F;
    stepFactor  = 0.5F;
}

//将neg.txt文件内图片名以地址形式存储到容器内，这个地址形式可以是绝对地址也可以是相对地址，要视具体情况而定
//当neg.txt和原始负样本图像在同一个文件夹（文件夹和traincascade.exe在同一级目录下）下，则使用相对地址，
//当neg.txt和原始负样本图像在同一个文件夹且neg.txt和traincascade.exe在同一级目录下，则使用绝对地址
bool CvCascadeImageReader::NegReader::create( const string _filename, Size _winSize )
{
    string dirname, str;
    std::ifstream file(_filename.c_str());
    if ( !file.is_open() )	//等价于if( !file )
        return false;

    size_t pos = _filename.rfind('\\');		//找不到时，返回string::npos
    char dlmrt = '\\';
    if (pos == string::npos)
    {
        pos = _filename.rfind('/');
        dlmrt = '/';
    }
    dirname = pos == string::npos ? "" : _filename.substr(0, pos) + dlmrt;	//substr(pos, n)拷贝字符串中从pos开始的n个字符
    while( !file.eof() )
    {
        std::getline(file, str);
        if (str.empty()) break;
        if (str.at(0) == '#' ) continue;	//neg.txt文件内可用以#开头进行注释(comment)
        imgFilenames.push_back(dirname + str);
    }
    file.close();

    winSize = _winSize;
    last = round = 0;
    return true;
}

bool CvCascadeImageReader::NegReader::nextImg()
{
    Point _offset = Point(0,0);
    size_t count = imgFilenames.size();	//查询得到neg.txt中共收录了多少张图片
    for( size_t i = 0; i < count; i++ )
    {
        src = imread( imgFilenames[last++], 0 );	//从头开始读取黑白图像
        if( src.empty() )
            continue;
        round += last / count;	//round用来记录现在是第几遍处理
        round = round % (winSize.width * winSize.height);
        last %= count;	//表示会一遍一遍的处理原始负样本，但每遍处理的不是同一个位置

        _offset.x = std::min( (int)round % winSize.width, src.cols - winSize.width );	//round % winSize.width取值空间在[0, winSize.width]
        _offset.y = std::min( (int)round / winSize.width, src.rows - winSize.height );	//round / winSize.width取值空间在[0, winSize.height]
        if( !src.empty() && src.type() == CV_8UC1
                && _offset.x >= 0 && _offset.y >= 0 )
            break;
    }

    if( src.empty() )
        return false; // no appropriate image
    point = offset = _offset;
    scale = max( ((float)winSize.width + point.x) / ((float)src.cols),
                 ((float)winSize.height + point.y) / ((float)src.rows) );

    Size sz( (int)(scale*src.cols + 0.5F), (int)(scale*src.rows + 0.5F) );	//加上0.5相当于四舍五入
    resize( src, img, sz );	//缩放原始负样本到sz这个尺寸大小
    return true;
}

bool CvCascadeImageReader::NegReader::get( Mat& _img )
{
    CV_Assert( !_img.empty() );
    CV_Assert( _img.type() == CV_8UC1 );
    CV_Assert( _img.cols == winSize.width );
    CV_Assert( _img.rows == winSize.height );

    if( img.empty() )	//适用于初始过程中img尚未初始化的情况
        if ( !nextImg() )
            return false;

    Mat mat( winSize.height, winSize.width, CV_8UC1,
        (void*)(img.data + point.y * img.step + point.x * img.elemSize()), img.step );
    mat.copyTo(_img);	//此时在_img内在原始负样本上取得到了一块窗口大小的训练用负样本

    if( (int)( point.x + (1.0F + stepFactor ) * winSize.width ) < img.cols )	//stepFactor为常量0.5F;
        point.x += (int)(stepFactor * winSize.width);
    else
    {
        point.x = offset.x;
        if( (int)( point.y + (1.0F + stepFactor ) * winSize.height ) < img.rows )
            point.y += (int)(stepFactor * winSize.height);
        else
        {
            point.y = offset.y;
            scale *= scaleFactor;
            if( scale <= 1.0F )
                resize( src, img, Size( (int)(scale*src.cols), (int)(scale*src.rows) ) );
            else
            {
                if ( !nextImg() )
                    return false;
            }
        }
    }
    return true;
}

CvCascadeImageReader::PosReader::PosReader()
{
    file = 0;
    vec = 0;
}

bool CvCascadeImageReader::PosReader::create( const string _filename )
{
    if ( file )
        fclose( file );
    file = fopen( _filename.c_str(), "rb" );	//以只读的方式打开一个二进制文件，打开失败则返回NULL

    if( !file )
        return false;
    short tmp = 0;
    if( fread( &count, sizeof( count ), 1, file ) != 1 ||			//vec文件内的正样本图像的个数
        fread( &vecSize, sizeof( vecSize ), 1, file ) != 1 ||		//vecSize=训练的命令参数(-w)*(-h)
        fread( &tmp, sizeof( tmp ), 1, file ) != 1 ||
        fread( &tmp, sizeof( tmp ), 1, file ) != 1 )				//测试使用，让指针file跳过开头(file是以字节为单位的)
        CV_Error_( CV_StsParseError, ("wrong file format for %s\n", _filename.c_str()) );
    base = sizeof( count ) + sizeof( vecSize ) + 2*sizeof( tmp );	//偏移的基址正好跳过开头部分
    if( feof( file ) )
        return false;
    last = 0;
    vec = (short*) cvAlloc( sizeof( *vec ) * vecSize );
    CV_Assert( vec );
    return true;
}

bool CvCascadeImageReader::PosReader::get( Mat &_img )
{
    CV_Assert( _img.rows * _img.cols == vecSize );	//确保opencv_traincascade.exe和opencv_createsamples.exe命令中的参数-w和-h一致
    uchar tmp = 0;
    size_t elements_read = fread( &tmp, sizeof( tmp ), 1, file );	//查看vec文件内是否还有图片
    if( elements_read != 1 )
        CV_Error( CV_StsBadArg, "Can not get new positive sample. The most possible reason is "
                                "insufficient count of samples in given vec-file.\n");
    elements_read = fread( vec, sizeof( vec[0] ), vecSize, file );	//fread的返回值为实际读取到数据的个数	//读取一张正样本
    if( elements_read != (size_t)(vecSize) )
        CV_Error( CV_StsBadArg, "Can not get new positive sample. Seems that vec-file has incorrect structure.\n");

    if( feof( file ) || last++ >= count )
        CV_Error( CV_StsBadArg, "Can not get new positive sample. vec-file is over.\n");

    for( int r = 0; r < _img.rows; r++ )
    {
        for( int c = 0; c < _img.cols; c++ )
            _img.ptr(r)[c] = (uchar)vec[r * _img.cols + c];	//_img的类型为CV_8UC1，	//_img.ptr(r)的作用是返回一个指向Mat举证第r行的指针
    }
    return true;
}

void CvCascadeImageReader::PosReader::restart()
{
    CV_Assert( file );
    last = 0;
    fseek( file, base, SEEK_SET );	//把文件指针的位置重置于_posFilename文件头偏移的base处，base = sizeof( count ) + sizeof( vecSize ) + 2*sizeof( tmp );，这正是正式图片信息前的引导部分大小
}

CvCascadeImageReader::PosReader::~PosReader()
{
    if (file)
        fclose( file );
    cvFree( &vec );
}
