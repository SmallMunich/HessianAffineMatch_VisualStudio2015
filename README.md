# HessianAffineMatch_VisualStudio2015

基于HessianAffine不变局部特征匹配算法

算法：Hessain多尺度仿射不变特征点提取 + 仿射变形参数估计构建SIFT描述符 + KnnMatch粗匹配 + RANSAC一致性提纯算法完成Two Views 匹配

版本：No.1 采取特征以文件.txt读入方式进行匹配
运行环境：Windows10 + Visual Studio2015 + OpenCV2.4.11 

依赖包已经动态设置，无需再次安装OpenCV! 如果运行缺少dll 则将bin文件下的dll拷贝至Debug模式下！


个人博客地址：
https://blog.csdn.net/small_munich/article/details/79639495 
