# 基于Python的图像边缘检测算法    
程序基于numpy实现       
### 算子     
程序包含五个内置图像边缘检测算子：Prewitt Sobel Laplace simpleLaplace canny       
其中前四个算子的mask为设定好的值，canny算法为手动实现     
### 依赖环境
numpy：核心计算环境      
PIL、OpenCV：提供打开图像和图像转矩阵的基本方法      
math：提供canny需要的数学常量          
copy：提供对象拷贝方法         
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
### 内置算子
prewitt_x : 为prewitt在X方向的operator(numpy.array)        
prewitt_y : 为prewitt在Y方向的operator(numpy.array)            
sobel_x : 为sobel在X方向的operator(numpy.array)        
sobel_y : 为sobel在Y方向的operator(numpy.array)                     
laplace : 为laplace算子(numpy.array)         
simpleLaplace : 为简易laplace算子(numpy.array)        
canny : 需要使用getGaussianMarix获取高斯核卷积后调用cannyKernel和cannyFinal来进行计算             
### 配套博客     
http://blog.accut.cn/archives/420           （如果提示404那就是我还没有写完）QAQ               