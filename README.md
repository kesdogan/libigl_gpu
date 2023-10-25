## Implementation of libigl `C++` methods using GPU via PyTorch

Several [libigl](https://libigl.github.io/) methods are very useful for geometric and graphics calculations. However, they are implemented for CPU and are thus slow to use on large problems. This repository presents some select methods implemented via PyTorch, which makes use of the Cuda architecture and provides a very efficient GPU implementation. 

The methods are not as precise and well-implemented as libigl (error of <0.1% approx), but serve the purpose, if maximum precision is not necessary. 

Enjoy! 
