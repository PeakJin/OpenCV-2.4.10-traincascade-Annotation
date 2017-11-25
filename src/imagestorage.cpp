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

//��neg.txt�ļ���ͼƬ���Ե�ַ��ʽ�洢�������ڣ������ַ��ʽ�����Ǿ��Ե�ַҲ��������Ե�ַ��Ҫ�Ӿ����������
//��neg.txt��ԭʼ������ͼ����ͬһ���ļ��У��ļ��к�traincascade.exe��ͬһ��Ŀ¼�£��£���ʹ����Ե�ַ��
//��neg.txt��ԭʼ������ͼ����ͬһ���ļ�����neg.txt��traincascade.exe��ͬһ��Ŀ¼�£���ʹ�þ��Ե�ַ
bool CvCascadeImageReader::NegReader::create( const string _filename, Size _winSize )
{
    string dirname, str;
    std::ifstream file(_filename.c_str());
    if ( !file.is_open() )	//�ȼ���if( !file )
        return false;

    size_t pos = _filename.rfind('\\');		//�Ҳ���ʱ������string::npos
    char dlmrt = '\\';
    if (pos == string::npos)
    {
        pos = _filename.rfind('/');
        dlmrt = '/';
    }
    dirname = pos == string::npos ? "" : _filename.substr(0, pos) + dlmrt;	//substr(pos, n)�����ַ����д�pos��ʼ��n���ַ�
    while( !file.eof() )
    {
        std::getline(file, str);
        if (str.empty()) break;
        if (str.at(0) == '#' ) continue;	//neg.txt�ļ��ڿ�����#��ͷ����ע��(comment)
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
    size_t count = imgFilenames.size();	//��ѯ�õ�neg.txt�й���¼�˶�����ͼƬ
    for( size_t i = 0; i < count; i++ )
    {
        src = imread( imgFilenames[last++], 0 );	//��ͷ��ʼ��ȡ�ڰ�ͼ��
        if( src.empty() )
            continue;
        round += last / count;	//round������¼�����ǵڼ��鴦��
        round = round % (winSize.width * winSize.height);
        last %= count;	//��ʾ��һ��һ��Ĵ���ԭʼ����������ÿ�鴦��Ĳ���ͬһ��λ��

        _offset.x = std::min( (int)round % winSize.width, src.cols - winSize.width );	//round % winSize.widthȡֵ�ռ���[0, winSize.width]
        _offset.y = std::min( (int)round / winSize.width, src.rows - winSize.height );	//round / winSize.widthȡֵ�ռ���[0, winSize.height]
        if( !src.empty() && src.type() == CV_8UC1
                && _offset.x >= 0 && _offset.y >= 0 )
            break;
    }

    if( src.empty() )
        return false; // no appropriate image
    point = offset = _offset;
    scale = max( ((float)winSize.width + point.x) / ((float)src.cols),
                 ((float)winSize.height + point.y) / ((float)src.rows) );

    Size sz( (int)(scale*src.cols + 0.5F), (int)(scale*src.rows + 0.5F) );	//����0.5�൱����������
    resize( src, img, sz );	//����ԭʼ��������sz����ߴ��С
    return true;
}

bool CvCascadeImageReader::NegReader::get( Mat& _img )
{
    CV_Assert( !_img.empty() );
    CV_Assert( _img.type() == CV_8UC1 );
    CV_Assert( _img.cols == winSize.width );
    CV_Assert( _img.rows == winSize.height );

    if( img.empty() )	//�����ڳ�ʼ������img��δ��ʼ�������
        if ( !nextImg() )
            return false;

    Mat mat( winSize.height, winSize.width, CV_8UC1,
        (void*)(img.data + point.y * img.step + point.x * img.elemSize()), img.step );
    mat.copyTo(_img);	//��ʱ��_img����ԭʼ��������ȡ�õ���һ�鴰�ڴ�С��ѵ���ø�����

    if( (int)( point.x + (1.0F + stepFactor ) * winSize.width ) < img.cols )	//stepFactorΪ����0.5F;
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
    file = fopen( _filename.c_str(), "rb" );	//��ֻ���ķ�ʽ��һ���������ļ�����ʧ���򷵻�NULL

    if( !file )
        return false;
    short tmp = 0;
    if( fread( &count, sizeof( count ), 1, file ) != 1 ||			//vec�ļ��ڵ�������ͼ��ĸ���
        fread( &vecSize, sizeof( vecSize ), 1, file ) != 1 ||		//vecSize=ѵ�����������(-w)*(-h)
        fread( &tmp, sizeof( tmp ), 1, file ) != 1 ||
        fread( &tmp, sizeof( tmp ), 1, file ) != 1 )				//����ʹ�ã���ָ��file������ͷ(file�����ֽ�Ϊ��λ��)
        CV_Error_( CV_StsParseError, ("wrong file format for %s\n", _filename.c_str()) );
    base = sizeof( count ) + sizeof( vecSize ) + 2*sizeof( tmp );	//ƫ�ƵĻ�ַ����������ͷ����
    if( feof( file ) )
        return false;
    last = 0;
    vec = (short*) cvAlloc( sizeof( *vec ) * vecSize );
    CV_Assert( vec );
    return true;
}

bool CvCascadeImageReader::PosReader::get( Mat &_img )
{
    CV_Assert( _img.rows * _img.cols == vecSize );	//ȷ��opencv_traincascade.exe��opencv_createsamples.exe�����еĲ���-w��-hһ��
    uchar tmp = 0;
    size_t elements_read = fread( &tmp, sizeof( tmp ), 1, file );	//�鿴vec�ļ����Ƿ���ͼƬ
    if( elements_read != 1 )
        CV_Error( CV_StsBadArg, "Can not get new positive sample. The most possible reason is "
                                "insufficient count of samples in given vec-file.\n");
    elements_read = fread( vec, sizeof( vec[0] ), vecSize, file );	//fread�ķ���ֵΪʵ�ʶ�ȡ�����ݵĸ���	//��ȡһ��������
    if( elements_read != (size_t)(vecSize) )
        CV_Error( CV_StsBadArg, "Can not get new positive sample. Seems that vec-file has incorrect structure.\n");

    if( feof( file ) || last++ >= count )
        CV_Error( CV_StsBadArg, "Can not get new positive sample. vec-file is over.\n");

    for( int r = 0; r < _img.rows; r++ )
    {
        for( int c = 0; c < _img.cols; c++ )
            _img.ptr(r)[c] = (uchar)vec[r * _img.cols + c];	//_img������ΪCV_8UC1��	//_img.ptr(r)�������Ƿ���һ��ָ��Mat��֤��r�е�ָ��
    }
    return true;
}

void CvCascadeImageReader::PosReader::restart()
{
    CV_Assert( file );
    last = 0;
    fseek( file, base, SEEK_SET );	//���ļ�ָ���λ��������_posFilename�ļ�ͷƫ�Ƶ�base����base = sizeof( count ) + sizeof( vecSize ) + 2*sizeof( tmp );����������ʽͼƬ��Ϣǰ���������ִ�С
}

CvCascadeImageReader::PosReader::~PosReader()
{
    if (file)
        fclose( file );
    cvFree( &vec );
}
