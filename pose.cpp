#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <string>
#include <iostream>
#include <sstream>
#include <cmath>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>

//!在动捕发送端使用ROS订阅相机的位置信息，并将其发送到Socket

// 创建一个全局的互斥锁
std::mutex flagMutex;
//!目前需要在这里手动选择，这个决定了动捕中订阅的刚体数量
int camera_num = 2;

// Socket通信相关参数
const char* server_ip = "10.106.11.116";
const int server_port1 = 60000;
const int server_port2 = 60001;

std::mutex socketMutex0;
std::mutex socketMutex1;

// 创建一个全局的计数器数组， counter的目的是降频接受频率
std::vector<int> counters(camera_num, 0);
// 发送标志位，发送了之后就置1
std::vector<int> sentflag(camera_num, 0);
// 累计标志位，记录该目标数据接收的次数，累计3次接收数据就置1，可以发送了
std::vector<int> accumlist(camera_num, 0);

std::string message_0 = std::to_string(0);
std::string message_1 = std::to_string(1);

// 函数原型声明
int initSocketConnection(int server_port);
void closeSocketConnection(int socket_fd);
void checkAllFlags();

void cameraPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& pose, int socket_fd, int now_i);

int main(int argc, char *argv[]) {
    setlocale(LC_ALL, "");

    // 初始化ROS节点
    ros::init(argc, argv, "get_camera_pose_node");
    ros::NodeHandle n;

    // 初始化Socket连接，这是第一个端口
    int serverSocket1 = initSocketConnection(server_port1);
    // 初始化Socket连接，这是第二个端口
    int serverSocket2 = initSocketConnection(server_port2);

    //TODO:要让多个刚体的消息都能订阅到，同时需要将订阅的消息放到一个数组里面，让socket每次发消息都能够一次把所有启用刚体的消息都发出去
    // 订阅VRPN消息
    ros::Subscriber sub_list[2];
    // int counter = 0;  // 计数器
    for (int i = 0; i < camera_num; i++) {
        std::string camera_name = "/vrpn_client_node/Cam_" + std::to_string(i) + "/pose";
        ROS_INFO("id: %d", i);
        if (i == 0)
        {
            sub_list[i] = n.subscribe<geometry_msgs::PoseStamped>(camera_name, 30, [serverSocket1, i](const geometry_msgs::PoseStamped::ConstPtr& pose) {
                cameraPoseCallback(pose, serverSocket1, i);
            });
        }
        else if (i == 1) {
            sub_list[i] = n.subscribe<geometry_msgs::PoseStamped>(camera_name, 30, [serverSocket2, i](const geometry_msgs::PoseStamped::ConstPtr& pose) {
                cameraPoseCallback(pose, serverSocket2, i);
            });
        }
        
    }

    // 进入ROS事件循环
    ros::spin();

    // 关闭Socket连接
    closeSocketConnection(serverSocket1);
    closeSocketConnection(serverSocket2);

    return 0;
}

// 初始化Socket连接
int initSocketConnection(int server_port) {
    int serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket < 0) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    // 第一个端口对应的ip目标
    struct sockaddr_in serverAddress;
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_addr.s_addr = inet_addr(server_ip);
    serverAddress.sin_port = htons(server_port);

    if (connect(serverSocket, (struct sockaddr*)&serverAddress, sizeof(serverAddress)) < 0) {
        perror("Connect failed");
        exit(EXIT_FAILURE);
    }

    return serverSocket;
}

// 关闭Socket连接
void closeSocketConnection(int socket_fd) {
    close(socket_fd);
}

// 检查所有标志位的函数，如果全都发送了，就将所有标志位都置0。用于保证发送消息的时序不变
void checkAllFlags() {
    std::lock_guard<std::mutex> lock(flagMutex);
    for (int i = 0; i < sentflag.size(); i++) {
        if (sentflag[i] == 0) {
            return;
        }
    }

    // 如果所有的标志位都为1，那么就将所有的标志位都置0
    for (int i = 0; i < sentflag.size(); i++) {
        sentflag[i] = 0;
    }
    ROS_INFO("所有标志位都为1，已经全部置0");
}

// 处理每个相机话题的回调函数
void cameraPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& pose, int socket_fd, int now_i) {
    
    float x = pose->pose.position.x;
    float y = pose->pose.position.y;
    float z = pose->pose.position.z;
    float ox = pose->pose.orientation.x;
    float oy = pose->pose.orientation.y;
    float oz = pose->pose.orientation.z;
    float ow = pose->pose.orientation.w;

    // ROS_INFO("counter now: %d", counters[now_i]);
    
    // 检查 x, y, z 是否都是有效的浮点数
    if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z))
    { 

        // 这一步是记录频率。并且只在sentflag为0的时候进入条件语句执行发送消息的操作
        if (counters[now_i] == 0 ) {
            ROS_INFO("now_i: %d", now_i); 
            ROS_INFO("position_x: %lf", x);
            ROS_INFO("position_y: %lf", y);
            ROS_INFO("position_z: %lf", z);
            ROS_INFO("orientation_x: %lf", ox);
            ROS_INFO("orientation_y: %lf", oy);
            ROS_INFO("orientation_z: %lf", oz);
            ROS_INFO("orientation_w: %lf", ow);
           
            // 将当前相机的位置信息存入messge中
            std::string message = std::to_string(x) + " " + std::to_string(y) + " " + std::to_string(z)+ " " + std::to_string(ox)+ " " + std::to_string(oy)+ " " + std::to_string(oz)+ " " + std::to_string(ow);
            switch (now_i) {
                case 0:
                    message_0 = message_0 + " " + message;
                    ROS_INFO("message_0: %s", message_0.c_str());
                    break;
                case 1:
                    message_1 = message_1 + " " + message;
                    ROS_INFO("message_1: %s", message_1.c_str());
                    break;
                default:
                    break;
            }

            if (true)
            {
                switch (now_i)
                {
                    case 0:
                    {
                        // 创建互斥锁
                        // std::unique_lock<std::mutex> lock(socketMutex0);
                        // std::lock_guard<std::mutex> lock(socketMutex0);
                        //*使用Socket发送数据
                        send(socket_fd, message_0.c_str(), message_0.size(), 0);
                        message_0 = std::to_string(0);
                        ROS_INFO("发送了message_0");
                        // 手动解锁互斥锁
                        // lock.unlock();
                        break;
                    }
                    case 1:
                    {
                        // 创建互斥锁
                        // std::unique_lock<std::mutex> lock(socketMutex1);
                        // std::lock_guard<std::mutex> lock(socketMutex1);
                        //*使用Socket发送数据
                        send(socket_fd, message_1.c_str(), message_1.size(), 0);
                        message_1 = std::to_string(1);
                        ROS_INFO("发送了message_1");
                        // 手动解锁互斥锁
                        // lock.unlock();
                        break;
                    }
                    default:
                        ROS_INFO("Invalid now_i: %d", now_i);
                        break;
                }

                accumlist[now_i] = 0;

                {
                    std::lock_guard<std::mutex> lock(flagMutex);
                    // sentflag[now_i] = 1;    //发送完毕后，将当前目标标志位置1
                }
            }
        }

        counters[now_i] = (counters[now_i] + 1) % 10;       //发送频率降频30倍，但信息一次发送3帧
    }
    else {
        ROS_WARN("Invalid pose data: x, y, z", x, y, z);
    }
}


