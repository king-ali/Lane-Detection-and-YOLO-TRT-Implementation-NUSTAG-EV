#include "class_timer.hpp"
#include "class_detector.h"
#include "opencv2/opencv.hpp"

#include <memory>
#include <thread>

using namespace cv;
int main()
{
	Config config_v3;
	config_v3.net_type = YOLOV3;
	config_v3.file_model_cfg = "../configs/yolov3.cfg";
	config_v3.file_model_weights = "../configs/yolov3.weights";
	config_v3.calibration_image_list_file_txt = "../configs/calibration_images.txt";
	config_v3.inference_precison =FP32;
	config_v3.detect_thresh = 0.5;

	Config config_v3_tiny;
	config_v3_tiny.net_type = YOLOV3_TINY;
	config_v3_tiny.detect_thresh = 0.7;
	config_v3_tiny.file_model_cfg = "../configs/yolov3-tiny.cfg";
	config_v3_tiny.file_model_weights = "../configs/yolov3-tiny.weights";
	config_v3_tiny.calibration_image_list_file_txt = "../configs/calibration_images.txt";
	config_v3_tiny.inference_precison = FP32;

	Config config_v4;
	config_v4.net_type = YOLOV4;
	config_v4.file_model_cfg = "../configs/yolov4.cfg";
	config_v4.file_model_weights = "../configs/yolov4.weights";
	config_v4.calibration_image_list_file_txt = "../configs/calibration_images.txt";
	config_v4.inference_precison = FP32;
	config_v4.detect_thresh = 0.5;

	Config config_v4_tiny;
	config_v4_tiny.net_type = YOLOV4_TINY;
	config_v4_tiny.detect_thresh = 0.5;
	config_v4_tiny.file_model_cfg = "../configs/yolov4-tiny.cfg";
	config_v4_tiny.file_model_weights = "../configs/yolov4-tiny.weights";
	config_v4_tiny.calibration_image_list_file_txt = "../configs/calibration_images.txt";
	config_v4_tiny.inference_precison = FP32;

	Config config_v5;
	config_v5.net_type = YOLOV5;
	config_v5.detect_thresh = 0.5;
	config_v5.file_model_cfg = "../configs/yolov5-5.0/yolov5s6.cfg";
	config_v5.file_model_weights = "../configs/yolov5-5.0/yolov5s6.weights";
	config_v5.calibration_image_list_file_txt = "../configs/calibration_images.txt";
	config_v5.inference_precison = FP32;

	std::unique_ptr<Detector> detector(new Detector());
	detector->init(config_v4_tiny);
	 Mat image0 =  imread("../configs/dur6.jpg",  IMREAD_UNCHANGED);
	 Mat image1 =  imread("../configs/person.jpg",  IMREAD_UNCHANGED);
	std::vector<BatchResult> batch_res;
	Timer timer;
	int deviceID = 0;             // 0 = open default camera
    int apiID =  CAP_ANY;     
	//VideoCapture cap("device=/dev/video1");
	//int cap_id =  CAP_GSTREAMER;
	std::cout <<"Here1";
	//VideoCapture cap("/dev/video0") ;
	//VideoCapture cap(1);

	// 0 = autodetect default API
    // open selected camera using selected API
    //cap.open(deviceID, apiID);
	//if(!cap.isOpened()){
	//	std::cout <<"Check camera";
	//	return -1;
	//}
	for (;;)
	{
		//prepare batch data
		std::vector< Mat> batch_img;
		Mat frame;
		//cap.read(frame);
		//if (frame.empty())
		//{	std::cout <<"Empty Frame";
		//	break;
		//}
		std::cout <<"In Loop";
		Mat temp0 = image0.clone();
		// Mat temp1 = image1.clone();
		batch_img.push_back(temp0);
		//batch_img.push_back(temp1);

		//detect
		timer.reset();
		detector->detect(batch_img, batch_res);
		timer.out("detect");

		//disp
		for (int i=0;i<batch_img.size();++i)
		{
			for (const auto &r : batch_res[i])
			{
				std::cout <<"batch "<<i<< " id:" << r.id << " prob:" << r.prob << " rect:" << r.rect << std::endl;
				 rectangle(batch_img[i], r.rect,  Scalar(255, 0, 0), 2);
				std::stringstream stream;
				stream << std::fixed << std::setprecision(2) << "id:" << r.id << "  score:" << r.prob;
				 putText(batch_img[i], stream.str(),  Point(r.rect.x, r.rect.y - 5), 0, 0.5,  Scalar(0, 0, 255), 2);
			}
			 namedWindow("image" + std::to_string(i),  WINDOW_NORMAL);
			 imshow("image"+std::to_string(i), batch_img[i]);
		}
		 waitKey(10);
	}
}
