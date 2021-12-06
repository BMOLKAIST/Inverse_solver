cd('/data01/Herve/code/_multiple_scattering_mshh/CUDA_CODE');

%mex -R2018a -output flir_cam_mex main.cpp CudaBorn.cpp CudaBorn.cpp '-IC:\Program Files\FLIR Systems\Spinnaker\include' '-LC:\Program Files\FLIR Systems\Spinnaker\lib64\vs2015' -lSpinnaker_v140;
mexcuda -R2018a -output CudaBorn main.cpp mex_utility.cpp CudaBorn.cpp memory_management_cuda.cu execution.cu execution2.cu  ...
    '-L/usr/local/cuda-11.2/lib64' -lcufft -DFFT_CUDA NVCC_FLAGS=--use_fast_math COMPFLAGS='$COMPFLAGS -std=c++17';



