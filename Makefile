MATLAB_DIR=/opt/local/MATLAB/R2017a

Atimesb:
	nvcc -arch=sm_60 -c Atimesb.cu -Xcompiler -fPIC -I$(MATLAB_DIR)/extern/include
	$(MATLAB_DIR)/bin/mex -L/usr/local/cuda-8.0/lib64 -lcudart -lcublas Atimesb.o
