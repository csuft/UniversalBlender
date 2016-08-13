Universal Blender
=================
This project is created to implement a universal blender for panorama images and videos by taking the advantage of parallel computing technologies(CUDA/OpenCL). The blender prefers to use CUDA rather than CPU or OpenCL. Then it will try to use OpenCL if there is no CUDA device to use. 

Advantages
----------
* Global single instance pattern.
* Support RGB(RGBA), BGR(BGRA) color model
* Use parallel computing underlying if possible
* Support arbitrary input resolution and output resolution
* Utilize log information to help debug
* Enough stable to use 24x7
* Offset can be replaced in any frequency. 
* Good inheritance hierarchy

How to compile?
---------------
* Configure OpenCL(to be continue)
* Configure CUDA(to be continue)

How to use?
-----------
The UniversalBlender can be used in any circumstances. The following code samples demonstrate how can we integrate UniversalBlender with our existing projects:  

    #include <iostream>  
	using namespace std;

	int main()  
	{  
		cout << "Hello World!" << endl;  
		return 0;  
	}  

License
-------
All copyright reserved. Arashi Vision Ltd. 2016. 
