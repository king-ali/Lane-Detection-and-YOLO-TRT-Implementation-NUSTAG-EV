#include "class_timer.hpp"
#include "class_detector.h"
#include "opencv2/opencv.hpp"
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



class Signal_Det {
    private:
	//image_transport::Subscriber sub;

	std::vector<BatchResult> batch_res;
	Timer timer;
	std::vector< Mat> batch_img;
	ros::Subscriber sub;
	ros::Publisher signal_pub;
	Config config_v3_tiny;
	std::queue<sensor_msgs::ImageConstPtr> img0_buf;
	Detector *detector;	
	std::stringstream ss;
	std_msgs::String signal;
	
	Mat image0;
	public:
    Signal_Det(ros::NodeHandle *nh) {
			
	
		sub = nh->subscribe("/usb_cam/image_raw", 1, &Signal_Det::img0_callback, this);
		signal_pub = nh->advertise<std_msgs::String>("/signal_info", 10);   
		config_v3_tiny.net_type = YOLOV3_TINY;
		config_v3_tiny.detect_thresh = 0.5;
		config_v3_tiny.file_model_cfg = "/VISION-TAG/detection-trt/detection-trt/configs/yolov3-tiny-bosch.cfg";
		config_v3_tiny.file_model_weights = "/VISION-TAG/detection-trt/detection-trt/configs/yolov3-tiny-bosch_40000.weights";
		config_v3_tiny.calibration_image_list_file_txt = "/VISION-TAG/detection-trt/detection-trt/configs/calibration_images.txt";
		config_v3_tiny.inference_precison = FP32;
		detector = new Detector();
		detector->init(config_v3_tiny);
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
				ss << r.id << ":" << r.prob << ":"  << r.rect.x << ":" << r.rect.y << ":" << r.rect.width<< ":" << r.rect.height <<std::endl;
				signal.data = ss.str();
			
				ROS_INFO("%s", signal.data.c_str());
				signal_pub.publish(signal);
				signal.data = "";
			}

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

	
	
	ros::init(argc, argv, "signal_detector");
    ros::NodeHandle nh("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
	Signal_Det signal = Signal_Det(&nh);

	
	ros::spin();

		
		
	
}
	
