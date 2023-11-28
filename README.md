## Implementation of libigl `c++` methods using GPU 

Several [libigl](https://libigl.github.io/) methods are very useful for geometric and graphics calculations. However, they are implemented for CPU and are thus slow to use on large problems. This repository presents some select methods implemented via PyTorch (and in the future, hopefully, directly via Cuda), which makes use of the Cuda architecture and provides a very efficient GPU implementation. 

The methods are not as precise and well-implemented as libigl (error of <0.1% approx), but serve the purpose, if maximum precision is not necessary. 

Enjoy! 


PyTorch: \
`signed_distance` &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; [signed_distance.py](https://github.com/tlk13/libigl_gpu/blob/main/pytorch_implementation/signed_distance.py) \
`per_face_normals` &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; [per_face_normals.py](https://github.com/tlk13/libigl_gpu/blob/main/pytorch_implementation/per_face_normals.py)\
`winding_number` &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; [winding_number.py](https://github.com/tlk13/libigl_gpu/blob/main/pytorch_implementation/winding_number.py)\
`snap_points` &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; [snap_points.py](https://github.com/tlk13/libigl_gpu/blob/main/pytorch_implementation/snap_points.py)