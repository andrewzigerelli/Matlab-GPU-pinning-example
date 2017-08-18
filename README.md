# Matlab-GPU-pinning-example
Calculate Ab where A is pinned to GPU for first call, b is sent repeatedly

To compile:
Edit MATLAB_DIR in the makefile
Also make sure the cuda include director is correct.
Then just run make.

Example:
A=rand(100,100); y=rand(100,1); orig_y=y;
for i=1:4
y=Atimesb(A,y);
end
ans_y=A^4*orig_y;
norm(y-ans_y)

