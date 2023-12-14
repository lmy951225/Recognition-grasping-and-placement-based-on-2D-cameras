# 本工程包含两部分，分别为2D相机眼在手上/手外的标定以及夹爪到tcp转换矩阵的标定

# 1.2D相机眼在手上/手外
   ## 1.1 拍照N张棋盘格图片，图片命名格式为img-n.jpg,图片保存为文件夹,记录机械臂tcp位姿,保存为txt文件
   ## 1.2 带入calib_handeye/calibration_all.py计算并得到各个位姿文件
   ![Alt text](camera_calibration.png)
   ## 1.3 将各个位姿文件带入calib_handeye/calibration_error.py计算得到各个真实点与计算点的误差
   ![Alt text](camera_error.png)
    
# 2.夹爪到tcp转换矩阵
   ## 2.1 按照给定顺序记录6个点的位姿，生成txt文件
   ## 2.2 带入calib_griper/griper2end.py计算得到转换矩阵
   ![Alt text](griper_calibration.png)
    
