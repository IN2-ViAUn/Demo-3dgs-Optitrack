# 基于动捕相机的3d高斯位姿编辑与渲染

> 本方法实现了在乐高场地内对3d高斯的小车进行实时位置更新。大体思路是用动捕相机获得小车的位置和朝向，发送到服务器进行处理之后，对3dgs小车进行编辑（小车和场地是分开的）然后渲染可视化，最终实现的效果是：在展示界面中，模型小车跟随实物小车在场地中进行运动。文档第一版：2024/10/5

## 1 平台及通讯

- 动捕相机的消息类型是VRPN（Virtual-Reality Peripheral Network），是一个用于虚拟现实和人机交互的开源软件库，提供了一种标准化的方式来连接和管理多种输入设备，支持实时数据流。
- 动捕相机的软件（Motive）在 windows 系统上，并且 VRPN 最好是用 ROS1 去获取，但是实验室的服务器全部都是 ubuntu22.04(ROS2)，因此需要在 windows 与 ubunu22.04 之间搭建一个信息传输的中转站。

最终的解决方案是：3090服务器上的 windows 上开一个 WLS（Windows Subsystem for Linux）版本的 ubuntu20.04 子系统，在子系统中完成对动捕相机消息的订阅和重发布，然后再用 4090 服务器（310的那一台）去订阅 WLS 通过 TCP 协议 发布的消息，最终是在4090服务器上完成小车位姿的处理以及渲染。
<div align="center">
![image](https://github.com/user-attachments/assets/41c27e4d-579a-4ec4-93be-6b332d55a0aa)
</div>
## 2 分支说明及注意事项

#### (1) main

- 主分支是在 4090 服务器上的代码，框架是天济师兄的 mini_gs，详见 ``G4DGS/G4DGS.py``。mini_gs 可以理解为把操作经典高斯以及调用 ``SIBR_remoteviewer`` 的代码单独摘出来，形成的一套能够便捷编辑3dgs球的代码，其中和渲染有关的属性拿出来定义了 ``MiniGaussian`` 类。
- 需要注意的是，在加载原版的3dgs点云时，需要调用 ``_original_3dgs_decoder`` 函数进行一定的转换。因为在原版高斯中，加载点云文件的方式就进行了 log 和 逆sigmoid 变换（ ``scene/gaussian_model.py/create_from_pcd函数``）。渲染之前需要再把它变回去，即取指数和做sigmoid变换。

```
scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
```

- 如果是对已经经过这个代码加载保存后的点云，则不需要进行二次解码，即之后不需要加"True"

```
gaussians_scene.init_scene_from_plyfile(os.path.join(args.workspace, 'field.ply'),True)
gs2 = gaussians_scene._load_ply(os.path.join(args.workspace, 'car_xe.ply'),True)
```

#### (2) 其他分支

- ``wsl_node 分支`` 是在 3090服务器windows系统的 WLS 上的代码
- ``mini_gs 分支`` 是天济师兄的 mini_gs 的代码，可以直接运行，用来学习框架

## 3 复现步骤

> 如果只是跑一下看看效果的话，可以跳过前两步模型准备，更改后的模型在 ``/data_xe``中，注意加载这两个点云文件不需要加“True”，因为已经不是原版高斯输出的点云了

#### (1) 乐高场地模型准备

- 使用经典的3dgs（改良的也行，后面只需要正确加载 .ply 文件即可）重建当前的乐高场地。步骤见 [CSDN复现3dgs教学](https://blog.csdn.net/weixin_45939751/article/details/136444065?spm=1001.2014.3001.5506)。为简化代码，最终方案是将场地坐标系(field)与动捕坐标系(capture)**手动对齐**，这样动捕相机发来的位姿就能直接使用，不需要做坐标系之间的变换。动捕坐标系见下图。
- 对齐方案使用的是CloudCompare。导入点云文件后，平移旋转点云至坐标系与动捕坐标系对齐，然后进行点云粗配准，步骤见 [CSDN点云配准教程](https://blog.csdn.net/qq_36686437/article/details/119966436?ops_request_misc=%257B%2522request%255Fid%2522%253A%25229D663390-24D3-4930-B575-5B0DE2DE3361%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=9D663390-24D3-4930-B575-5B0DE2DE3361&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-119966436-null-null.142^v100^pc_search_result_base4&utm_term=cloudcompare%E7%82%B9%E4%BA%91%E9%85%8D%E5%87%86%E8%9E%8D%E5%90%88&spm=1018.2226.3001.4187)。得到矩阵之后，需要运行 gs_save.py 以便将高斯球的方向也进行旋转，避免出现毛刺情况。使用本办法保存的点云，再次加载的时候就不需要二次解码（即加载点云时不需要加"True"）

<div align="center">
  <img width="350" alt="78cc4352fec7d7c68dd568f416e572b" src="https://github.com/user-attachments/assets/1e2897c2-6d3d-49ef-999b-57cb7839dd77">
</div>

#### (2) 单独小车模型准备

- 小车作为前景需要单独重建，但重建出来有周围环境的高斯点，我们需要用 CloudCompare 软件剔除。3D高斯是一种特殊的点云，所以在处理之前需要转化为cloudcompare支持的格式（以下简称.cc），方法：[GitHub-3d高斯与cc格式转化](https://github.com/francescofugazzi/3dgsconverter)，之后导入 cc 格式的点云后进行修建即可 [CSDN删除点云教程](https://blog.csdn.net/qq_45250951/article/details/125748474?ops_request_misc=%257B%2522request%255Fid%2522%253A%25224B40CB79-05D8-41A4-9C95-ADEA26755037%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=4B40CB79-05D8-41A4-9C95-ADEA26755037&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-4-125748474-null-null.142^v100^pc_search_result_base4&utm_term=cloudcompare%E5%8E%BB%E9%99%A4%E7%82%B9%E4%BA%91&spm=1018.2226.3001.4187)。之后保存小车点云后还需要转化为 3dgs格式。
- 需要注意的是，cloudcompare只能对3dgs的xyz属性进行操作，因此如果涉及到旋转会产生毛刺，这里只用它进行点云修剪。

#### (3) 坐标变换矩阵测算

- 动捕坐标系和场地坐标系存在变换关系，本方案采用的是最小二乘法（波翰），在场地内不同位置放置小车，用cloudcompare可以读出一个坐标（选取第二个动捕球根部的点近似为刚体中心），也会收到动捕的坐标。多个位置多次拟合出一个4*4变换矩阵，那么每次受到动捕发来的xyz，左乘这个矩阵即可得到场地坐标系下的小车位置。
- 测出坐标来只需要填到 RT_calculate.py 中，运行得到 ``动捕坐标系到场地坐标系``的变换矩阵。

#### (4) 启动程序步骤

- 初版代码设定能同时对两辆小车进行操作，因此需要打开 ``mini_bone_30000.py`` 和 ``mini_bone_30001.py``， wsl上的代码只有同时与两个端口建立联系后才会发出消息。
- 在 wsl 运行命令：``roslaunch vrpn_client_ros sample.launch server:=127.0.0.1``，因为第一次接受 ROS命令是在本机，所以ip地址为127.0.0.1
- 在 4090 服务器上运行 SIBR_remoteviewer
  ```
  cd yXe_file/3DGS/gaussian-splatting/SIBR_viewers/install/bin/
  # 默认端口为6009
  ./SIBR_remoteGaussian_app --ip 127.0.0.1 --port 6009 --path /home/wangyixian/yXe_file/3DGS/gaussian-splatting/data/lego
  ```
- mini_bone_30000.py 和 30001 如果正常输出日志（收到并计算出了此时的小车位置），但看着仍然是一片黑，则可以按Y切换到 trackball 模式滑动滚轮缩小视角，应该能够找到此时的场地。动一下小车就会发现场地上的小车模型也在跟着运动。
- 注意：白色小车对应的动捕刚体是Cam_0（端口号30000），绿色小车对应的是Cam_1（端口号30001）
