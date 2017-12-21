# 基于Python的图像边缘检测算法    
程序基于numpy实现       
### 算子     
程序包含五个内置图像边缘检测算子：Prewitt Sobel Laplace simpleLaplace canny       
其中前四个算子的mask为设定好的值，canny算法为手动实现     
### 依赖环境
numpy：核心计算环境      
PIL、OpenCV：提供打开图像和图像转矩阵的基本方法      
### 函数列表
openImg_opencv(filename = 'new.jpg') : 使用OpenCV打开图像并返回图像的numpy数组         
openImg_PIL(filename = 'new.jpg')  : 使用PIL打开图像并返回图像的numpy数组         
averageGray(sourceImage) : 将图像数组进行常规灰度化      
averageGrayWithWeighted(sourceImage) : 将图像数组进行带权灰度化        
maxGray(sourceImage) : 选用RGB值中最大的一个进行灰度化         
convolution(sourceImage, operator, size = 3) : 使用大小为size的operator对sourceImage进行卷积返回卷积结果            
getGaussianMarix(size = 3, padding = 1) : 获取大小为size内偏移为padding的高斯核（所谓的内偏移是为了生成二位高斯分布矩阵）          
cannyKernel(sourceImage) : canny算法核心，返回没有加强图像边缘识别结果          
cannyFinal(sourceImage) : 对cannyKernel的识别结果进行强化            