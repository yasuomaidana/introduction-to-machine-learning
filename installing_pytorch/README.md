## Notes
If you don't have any cuda compatibility just change
```requirements.txt
torch == 1.13.1+cu117
torchvision == 0.14.1+cu117
cuda-python
nvidia-pyindex
--extra-index-url https://pypi.ngc.nvidia.com
nvidia-cuda-runtime-cu12
```
by
```requirements.txt
torch
torchvision
```
As you notice if you are not using cuda, there is no reason to have all the cuda-python and nvidia libraries