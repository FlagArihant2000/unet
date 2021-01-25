# U - Net for Image Segmentation

![](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)

U - Net is a popular architecture used for semantic segmentation. It was first published with the aim of segmentation in medical images. It is also one of the initial architectures on semantic segmentation, that involves the use of an encoder - decoder framework.

An encoder - decoder framework is a useful concept in deep learning, that utilises the representation of high level features in lower dimensional space. The encoder network takes the image as input, followed by a series of convolutional layers and downsampling operations. It is from the low dimensional space obtained from encoder that the decoder framework is used for upsampling. A few convolutional layers are also used and are considered as correction stages after an upsampling is done.

In U - Net, the corresponding stage in encoder framework is copied and cropped onto the the decoder step. Intuitively, this represents the incorporation of lower level features that are obtained during the initial stages of convolutions.

Original Paper: [Link](https://arxiv.org/pdf/1505.04597.pdf)

Results for this particular repository, along with code execution steps will be added later.

