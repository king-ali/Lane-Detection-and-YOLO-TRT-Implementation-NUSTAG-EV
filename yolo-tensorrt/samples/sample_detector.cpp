#include "class_timer.hpp"
#include "class_detector.h"
#include "opencv2/opencv.hpp"
//#include <mutex>
#include <queue>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <memory>
#include <thread>
#include "std_msgs/String.h"
#include <sstream>

using namespace cv;

//std::mutex m_buf;



/*
void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{

	std::cout<<"Image callback"<<std::endl;
    //m_buf.lock();
    img0_buf.push(img_msg);
	namedWindow("image",  WINDOW_NORMAL);
	image0 = getImageFromMsg(img0_buf.front());
	img0_buf.pop();
	//imshow("image", image0);
	batch_img.push_back(image0);
	
	//m_buf.unlock();
}
*/

class Sign_Det {
    private:
	//image_transport::Subscriber sub;

	std::vector<BatchResult> batch_res;
	Timer timer;
	std::vector< Mat> batch_img;
	ros::Subscriber sub;
	ros::Publisher sign_pub;
	Config config_v4_tiny;
	std::queue<sensor_msgs::ImageConstPtr> img0_buf;
	Detector *detector;	
	std::stringstream ss;
	std_msgs::String sign;
	
	Mat image0;
	public:
    Sign_Det(ros::NodeHandle *nh) {
			
		//image_transport::ImageTransport it(nh);
		sub = nh->subscribe("/usb_cam/image_raw", 1, &Sign_Det::img0_callback, this);
		sign_pub = nh->advertise<std_msgs::String>("/sign_info", 1);   
		config_v4_tiny.net_type = YOLOV4_TINY;
		config_v4_tiny.detect_thresh = 0.5;
		config_v4_tiny.file_model_cfg = "/home/brain/MYNT-EYE-S-SDK/wrappers/ros/src/yolo-tensorrt/configs/new-yolov4-tiny-detector.cfg";
		config_v4_tiny.file_model_weights = "/home/brain/MYNT-EYE-S-SDK/wrappers/ros/src/yolo-tensorrt/configs/new-yolov4-tiny-detector.weights";
		config_v4_tiny.calibration_image_list_file_txt = "/home/brain/MYNT-EYE-S-SDK/wrappers/ros/src/yolo-tensorrt/configs/calibration_images.txt";
		config_v4_tiny.inference_precison = FP32;
		detector = new Detector();
		detector->init(config_v4_tiny);
		namedWindow("image",  WINDOW_NORMAL);
		
    }
    void img0_callback(const sensor_msgs::ImageConstPtr &img_msg) {
		//std::cout<<"Image callback"<<std::endl;
		//m_buf.lock();
		img0_buf.push(img_msg);
		image0 = this->getImageFromMsg(img0_buf.front());
		img0_buf.pop();
		//imshow("image", image0);
		//waitKey(10);
		//if (batch_img.empty()){
		batch_img.push_back(image0);
		detector->detect(batch_img, batch_res);
	
		
		if (!batch_img.empty()){
		for (int i=0;i<batch_img.size();++i)
		{
			for (const auto &r : batch_res[i])
			{
				std::cout <<"batch "<<i<< " id:" << r.id << " prob:" << r.prob << " rect:" << r.rect << std::endl;
					rectangle(batch_img[i], r.rect,  Scalar(255, 0, 0), 2);
				std::stringstream stream;
				stream << std::fixed << std::setprecision(2) << "id:" << r.id << "  score:" << r.prob;
					putText(batch_img[i], stream.str(),  Point(r.rect.x, r.rect.y - 5), 0, 0.5,  Scalar(0, 0, 255), 2);
				ss.str(std::string());
				ss << r.id << ":" << r.prob << ":"  << r.rect.x << ":" <<r.rect.y<<":" <<r.rect.width<<":" <<r.rect.height<<std::endl;
				sign.data = ss.str();
			
				ROS_INFO("%s", sign.data.c_str());
				sign_pub.publish(sign);
				sign.data = "";
			}
			std::cout<<sign.data<<std::endl;
			imshow("image", batch_img[i]);
				

		}
		batch_img.pop_back();
		
		waitKey(10);
		}
				
    }
	cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg){
		cv_bridge::CvImageConstPtr ptr;
		if (img_msg->encoding == "8UC1")
		{
			sensor_msgs::Image img;
			img.header = img_msg->header;
			img.height = img_msg->height;
			img.width = img_msg->width;
			img.is_bigendian = img_msg->is_bigendian;
			img.step = img_msg->step;
			img.data = img_msg->data;
			img.encoding = "bgr8";
			ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
		}
		else
			ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);

		Mat img = ptr->image.clone();
		return img;
	}

};

int main(int argc, char **argv){

	
	//detector->init(config_v4_tiny);
	//timer.reset();
	//detector->detect(batch_img, batch_res);
	//timer.out("detect");
		
	ros::init(argc, argv, "sign_detector");
    ros::NodeHandle nh("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
	Sign_Det sign = Sign_Det(&nh);

	
	ros::spin();
	/*
	Config config_v3;
	config_v3.net_type = YOLOV3;
	config_v3.file_model_cfg = "../configs/yolov3.cfg";
	config_v3.file_model_weights = "../configs/yolov3.weights";
	config_v3.calibration_image_list_file_txt = "../configs/calibration_images.txt";
	config_v3.inference_precison =FP32;
	config_v3.detect_thresh = 0.5;

	Config config_v3_tiny;
	config_v3_tiny.net_type = YOLOV3_TINY;
	config_v3_tiny.detect_thresh = 0.5;
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
	config_v4_tiny.file_model_cfg = "/home/brain/MYNT-EYE-S-SDK/wrappers/ros/src/yolo-tensorrt/configs/yolov4-tiny.cfg";
	config_v4_tiny.file_model_weights = "/home/brain/MYNT-EYE-S-SDK/wrappers/ros/src/yolo-tensorrt/configs/yolov4-tiny.weights";
	config_v4_tiny.calibration_image_list_file_txt = "/home/brain/MYNT-EYE-S-SDK/wrappers/ros/src/yolo-tensorrt/configs/calibration_images.txt";
	config_v4_tiny.inference_precison = FP32;

	Config config_v5;
	config_v5.net_type = YOLOV5;
	config_v5.detect_thresh = 0.5;
	config_v5.file_model_cfg = "../configs/yolov5-5.0/yolov5s6.cfg";
	config_v5.file_model_weights = "../configs/yolov5-5.0/yolov5s6.weights";
	config_v5.calibration_image_list_file_txt = "../configs/calibration_images.txt";
	config_v5.inference_precison = FP32;
	*/
	// Mat image0 =  imread("../configs/dur6.jpg",  IMREAD_UNCHANGED);
	// Mat image1 =  imread("../configs/person.jpg",  IMREAD_UNCHANGED);

	//ros::Subscriber sub_img0 = n.subscribe("/mynteye/right/image_raw", 10, img0_callback);
    
	//int deviceID = 0;             // 0 = open default camera
    //int apiID =  CAP_ANY;     
	//VideoCapture cap("device=/dev/video1");
	//int cap_id =  CAP_GSTREAMER;
	//std::cout <<"Here1";
	//VideoCapture cap("/dev/video0") ;
	//VideoCapture cap(1);

	// 0 = autodetect default API
    // open selected camera using selected API
    //cap.open(deviceID, apiID);
	//if(!cap.isOpened()){
	//	std::cout <<"Check camera";
	//	return -1;
	//}
	//for (;;)
	//{	
		//if (!img0_buf.empty() )
		//{
		//	std::cout<<"ROS Image"<<std::endl;
		//		image0 = getImageFromMsg(img0_buf.front());
		//		img0_buf.pop();
		
		
		//prepare batch data

		//Mat frame;
		//cap.read(frame);
		//if (frame.empty())
		//{	std::cout <<"Empty Frame";
		//	break;
		//}
		//std::cout <<"In Loop Image ROS"<<std::endl;
		//Mat temp0 = image0.clone();
		
		// Mat temp1 = image1.clone();
		//batch_img.push_back(temp1);

		//detect
		
		
	
}
	
