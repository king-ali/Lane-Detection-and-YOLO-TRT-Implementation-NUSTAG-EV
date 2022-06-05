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
		config_v4_tiny.file_model_cfg = "/VISION-TAG/detection-trt/detection-trt/configs/new-yolov4-tiny-detector.cfg";
		config_v4_tiny.file_model_weights = "/VISION-TAG/detection-trt/detection-trt/configs/new-yolov4-tiny-detector.weights";
		config_v4_tiny.calibration_image_list_file_txt = "/VISION-TAG/detection-trt/detection-trt/configs/calibration_images.txt";
		config_v4_tiny.inference_precison = FP32;
		detector = new Detector();
		detector->init(config_v4_tiny);
		namedWindow("image",  WINDOW_NORMAL);
		
    }
    void img0_callback(const sensor_msgs::ImageConstPtr &img_msg) {
		
		img0_buf.push(img_msg);
		image0 = this->getImageFromMsg(img0_buf.front());
		img0_buf.pop();
		
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


		
	ros::init(argc, argv, "sign_detector");
    ros::NodeHandle nh("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
	Sign_Det sign = Sign_Det(&nh);

	
	ros::spin();

		
		
	
}
	
