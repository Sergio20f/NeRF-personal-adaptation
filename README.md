# NeRF and Applications Overview

NeRF is a computer vision technique that generates novel views of complex scenes by optimizing an
underlying continuous volumetric function using a sparse set of input views. The continuous scene is
described as a 5D vector function containing information about the spatial location and view angle. It
optimises a deep fully connected multilayer perceptron (no convolution) through gradient descent by
minimising the error between the ground truth image and the corresponding views from the generated
render.
The intention of this document is to give a short but comprehensive overview on the NeRF method
and its inner workings. In term of applications, we will focus on trying to explore the possibility of
taking advantage of the NeRF scene synthesis capabilities to potential advancements in the field of
systems and control. The focus is towards developing an understanding on how the NeRF technique
might be used for online state updates in an autonomous system inside the same environment used
for training NeRF. As well, we will study the probable relationship between the NeRF outputs and
the current navigation planning algorithms available.
