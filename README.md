# SISR-Survey

### An investigation project for SISR.  <a href="https://arxiv.org/abs/2109.14335">[Paper]</a> 

### This repository is an official project of the paper "From Beginner to Master: A Survey for Deep Learning-based Single-Image Super-Resolution". 

## Purpose

<font color='red'>Due to the pages and time limitation, it is impossible to introduce all SISR methods in the paper, and it is impossible to update the latest methods in time. Therefore, we use this project to assist our survey to cover more methods. This will be a continuously updated project! We hope it can help more researchers and promote the development of image super-resolution. Welcome more researchers to jointly maintain this project!</font>

<p align="center">
<img src="Images/SISR.png" width="400px"/>
</p>

## Abstract

Single-image super-resolution (SISR) is an important task in image processing, which aims to enhance the resolution of imaging systems. Recently, SISR has made a huge leap and has achieved promising results with the help of deep learning (DL). In this survey, we give an overview of DL-based SISR methods and group them according to their targets, such as reconstruction efficiency, reconstruction accuracy, and perceptual accuracy. Specifically, we first introduce the problem definition, research background, and the significance of SISR. Secondly, we introduce some related works, including benchmark datasets, upsampling methods, optimization objectives, and image quality assessment methods. Thirdly, we provide a detailed investigation of SISR and give some domain-specific applications of it. Fourthly, we present the reconstruction results of some classic SISR methods to intuitively know their performance. Finally, we discuss some issues that still exist in SISR and summarize some new trends and future directions. This is an exhaustive survey of SISR, which can help researchers better understand SISR and inspire more exciting research in this field. 

## Taxonomy

<p align="center">
<img src="Images/Framework.png" width="1000px"/>
</p>

## Datasets

Benchmarks datasets for single-image super-resolution (SISR).

<p align="center">
<img src="Images/Datasets.png" width="1000px"/>
</p>
# SINGLE-IMAGE SUPER-RESOLUTION

## Reconstruction Efficiency Methods

## Perceptual Quality Methods

## Perceptual Quality Methods

## Further Improvement Methods

# DOMAIN-SPECIFIC APPLICATIONS

## Real-World SISR

The degradation modes are complex and unknown in real-world scenarios, where downsampling is usually performed after anisotropic blurring and sometimes signal-dependent noise is added.  Recently, some new technologies have been proposed, such as unsupervised learning, self-supervised learning, zero-shot learning, meta-learning, blind SISR, and scale arbitrary SISR. In this part, we introduce the latter three methods due to their impressive foresight and versatility.

### Blind SISR

[1] <a href="https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Learning_a_Single_CVPR_2018_paper.pdf">Learning A Single Convolutional Super-Resolution Network for Multiple Degradations</a> 

[2] <a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Deep_Plug-And-Play_Super-Resolution_for_Arbitrary_Blur_Kernels_CVPR_2019_paper.pdf">Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels</a> 

[3] <a href="https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_Unified_Dynamic_Convolutional_Network_for_Super-Resolution_With_Variational_Degradations_CVPR_2020_paper.pdf">Unified Dynamic Convolutional Network for Super-Resolution with Variational Degradations</a> 

[4] <a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Hui_Learning_the_Non-Differentiable_Optimization_for_Blind_Super-Resolution_CVPR_2021_paper.pdf">Learning the Non-Differentiable Optimization for Blind Super-Resolution</a> 

[5] <a href="https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Deep_Unfolding_Network_for_Image_Super-Resolution_CVPR_2020_paper.pdf">Deep Unfolding Network for Image Super-Resolution</a> 

[6] <a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Gu_Blind_Super-Resolution_With_Iterative_Kernel_Correction_CVPR_2019_paper.pdf">Blind Super-Resolution with Iterative Kernel Correction</a> 

[7] <a href="https://arxiv.org/pdf/2010.02631.pdf">Unfolding the Alternating Optimization for Blind Super Resolution</a> 

[8] <a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Kim_KOALAnet_Blind_Super-Resolution_Using_Kernel-Oriented_Adaptive_Local_Adjustment_CVPR_2021_paper.pdf">KOALAnet: Blind Super-Resolution using Kernel-Oriented Adaptive Local Adjustment</a> 

[9] <a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Kim_KOALAnet_Blind_Super-Resolution_Using_Kernel-Oriented_Adaptive_Local_Adjustment_CVPR_2021_paper.pdf">KernelNet: A Blind Super-Resolution Kernel Estimation Network</a> 

[10] <a href="https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w13/Yuan_Unsupervised_Image_Super-Resolution_CVPR_2018_paper.pdf">Unsupervised Image Super-Resolution Using Cycle-in-Cycle Generative Adversarial Networks</a>  

### Meta-Learning

[1] <a href="https://openaccess.thecvf.com/content_CVPR_2020/papers/Soh_Meta-Transfer_Learning_for_Zero-Shot_Super-Resolution_CVPR_2020_paper.pdf">Meta-Transfer Learning for Zero-Shot Super-Resolution</a> 

[2] <a href="https://link.springer.com/content/pdf/10.1007/978-3-030-58583-9_45.pdf">Fast Adaptation to Super-Resolution Networks via Meta-Learning</a> 

[3] <a href="https://ieeexplore.ieee.org/abstract/document/9180081">Meta-USR: A Unified Super-Resolution Network for Multiple Degradation Parameters</a> 

### Scale Arbitrary SISR

[1] <a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Hu_Meta-SR_A_Magnification-Arbitrary_Network_for_Super-Resolution_CVPR_2019_paper.pdf">Meta-SR: A Magnification-Arbitrary Network for Super-Resolution</a> 

[2] <a href="https://ieeexplore.ieee.org/abstract/document/9180081">Meta-USR: A Unified Super-Resolution Network for Multiple Degradation Parameters</a> 

[3] <a href="https://www.researchgate.net/profile/Longguang-Wang/publication/340523881_Learning_A_Single_Network_for_Scale-Arbitrary_Super-Resolution/links/60fc178b1e95fe241a85a2f9/Learning-A-Single-Network-for-Scale-Arbitrary-Super-Resolution.pdf">Learning A Single Network for Scale-Arbitrary Super-Resolution}</a> 

## Remote Sensing Image Super-Resolution

With the development of satellite image processing, remote sensing has become more and more important. However, due to the limitations of current imaging sensors and complex atmospheric conditions, such as limited spatial resolution, spectral resolution, and radiation resolution, we are facing huge challenges in remote sensing applications. 

[1] <a href="https://ieeexplore.ieee.org/abstract/document/8400496/">A New Deep Generative Network for Unsupervised Remote Sensing Single-Image Super-Resolution</a> 

[2] <a href="https://www.mdpi.com/2072-4292/11/15/1817">Deep Residual Squeeze and Excitation Network for Remote Sensing Image Super-Resolution</a> 

[3] <a href="https://ieeexplore.ieee.org/abstract/document/9151234">Remote Sensing Image Super-Resolution via Mixed High-order Attention Network</a> 

[4] <a href="https://ieeexplore.ieee.org/abstract/document/9194276">Remote Sensing Image Super-Resolution Using Second-Order Multi-Scale Networks</a> 

## Hyperspectral Image Super-Resolution

In contrast to human eyes that can only be exposed to visible light, hyperspectral imaging is a technique for collecting and processing information across the entire range of electromagnetic spectrum. The hyperspectral system is often compromised due to the limitations of the amount of the incident energy, hence there is a trade-off between the spatial and spectral resolution. Therefore, hyperspectral image super-resolution is studied to solve this problem.

[1] <a href="https://www.mdpi.com/2072-4292/9/11/1139">Hyperspectral Image Spatial Super-Resolution via 3D Full Convolutional Neural Network</a> 

[2] <a href="https://ieeexplore.ieee.org/abstract/document/8499097">Single Hyperspectral Image Super-Resolution with Grouped Deep Recursive Residual Network</a> 

[3] <a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Fu_Hyperspectral_Image_Super-Resolution_With_Optimized_RGB_Guidance_CVPR_2019_paper.pdf">Hyperspectral Image Super-Resolution with Optimized RGB Guidance</a> 

[4] <a href="https://ieeexplore.ieee.org/abstract/document/9097432">Learning Spatial-Spectral Prior for Super-Resolution of Hyperspectral Imagery</a> 

[5] <a href="https://ieeexplore.ieee.org/abstract/document/9329109">A Spectral Grouping and Attention-Driven Residual Dense Network for Hyperspectral Image Super-Resolution</a> 

## Light Field Image Super-Resolution

Light field (LF) camera is a camera that can capture information about the light field emanating from a scene and can provide multiple views of a scene. Recently, the LF image is becoming more and more important since it can be used for post-capture refocusing, depth sensing, and de-occlusion. However, LF cameras are faced with a trade-off between spatial and angular resolution. In order to solve this issue, SR technology is introduced to achieve a good balance between spatial and angular resolution.

[1] <a href="https://ieeexplore.ieee.org/abstract/document/7856946">Light-field Image Super-Resolution Using Convolutional Neural Network</a> 

[2] <a href="https://ieeexplore.ieee.org/abstract/document/8356655">LFNet: A novel Bidirectional Recurrent Convolutional Neural Network for Light-field Image Super-Resolution</a> 

[3] <a href="https://link.springer.com/chapter/10.1007/978-3-030-58592-1_18">Spatial-Angular Interaction for Light Field Image Super-Resolution</a> 

[4] <a href="https://ieeexplore.ieee.org/abstract/document/9286855">Light Field Image Super-Resolution Using Deformable Convolution</a> 

## Face Image Super-Resolution

Face image super-resolution is the most famous field in which apply SR technology to domain-specific images. Due to the potential applications in facial recognition systems such as security and surveillance, face image super-resolution has become an active area of research. 

[1] <a href="https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewFile/9752/9824">Learning Face Hallucination in the Wild</a> 

[2] <a href="https://link.springer.com/chapter/10.1007/978-3-319-46454-1_37">Deep Cascaded Bi-Network for Face Hallucination</a> 

[3] <a href="https://openaccess.thecvf.com/content_cvpr_2017/papers/Yu_Hallucinating_Very_Low-Resolution_CVPR_2017_paper.pdf">Hallucinating Very Low-Resolution Unaligned and Noisy Face Images by Transformative Discriminative Autoencoders</a> 

[4] <a href="https://openaccess.thecvf.com/content_ECCV_2018/papers/Kaipeng_Zhang_Super-Identity_Convolutional_Neural_ECCV_2018_paper.pdf">Super-Identity Convolutional Neural Network for Face Hallucination</a> 

[5] <a href="https://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Dogan_Exemplar_Guided_Face_Image_Super-Resolution_Without_Facial_Landmarks_CVPRW_2019_paper.pdf">Exemplar Guided Face Image Super-Resolution without Facial Landmarks</a> 

[6] <a href="https://dl.acm.org/doi/pdf/10.1145/3418462">Robust Facial Image Super-Resolution by Kernel Locality-Constrained Coupled-Layer Regression</a> 

## Medical Image Super-Resolution

Medical imaging methods such as computational tomography (CT) and magnetic resonance imaging (MRI) are essential to clinical diagnoses and surgery planning. Hence, high-resolution medical images are desirable to provide necessary visual information of the human body. Recently, many methods have been proposed for medical image super-resolution

[1] <a href="https://link.springer.com/content/pdf/10.1007%2F978-3-030-00928-1_11.pdf">Efficient and Accurate MRI Super-Resolution Using A Generative Adversarial Network and 3D Multi-Level Densely Connected Network</a> 

[2] <a href="https://www.sciencedirect.com/science/article/pii/S0098300418310562">CT-Image of Rock Samples Super Resolution Using 3D Convolutional Neural Network</a> 

[3] <a href="https://ieeexplore.ieee.org/abstract/document/8736987">Channel Splitting Network for Single MR Image Super-Resolution</a> 

[4] <a href="https://openaccess.thecvf.com/content_CVPR_2020/papers/Peng_SAINT_Spatially_Aware_Interpolation_NeTwork_for_Medical_Slice_Synthesis_CVPR_2020_paper.pdf">SAINT: Spatially Aware Interpolation Network for Medical Slice Synthesis</a> 

## Depth Map Super-Resolution

The depth map is an image or image channel that contains information relating to the distance of the surfaces of scene objects from a viewpoint. The use of depth information of a scene is essential in many applications such as autonomous navigation, 3D reconstruction, human-computer interaction, and virtual reality. However, depth sensors, such as Microsoft Kinect and Lidar, can only provide depth maps of limited resolutions. Hence, depth map super-resolution has drawn more and more attention recently. 

[1] <a href="https://link.springer.com/chapter/10.1007/978-3-319-54190-7_22">Deep Depth Super-Resolution: Learning Depth Super-Resolution Using Deep Convolutional Neural Network</a> 

[2] <a href="https://link.springer.com/chapter/10.1007/978-3-319-46487-9_17">Atgv-net: Accurate Depth Super-Resolution</a> 

[3] <a href="https://link.springer.com/chapter/10.1007/978-3-319-46487-9_22">Depth Map Super-Resolution by Deep Multi-Scale Guidance</a> 

[4] <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8443445">Deeply Supervised Depth Map Super-Resolution as Novel View Synthesis</a> 

[5] <a href="https://openaccess.thecvf.com/content_ICCV_2019/papers/Voynov_Perceptual_Deep_Depth_Super-Resolution_ICCV_2019_paper.pdf">Perceptual Deep Depth Super-Resolution</a> 

[6] <a href="https://openaccess.thecvf.com/content_CVPR_2020/papers/Song_Channel_Attention_Based_Iterative_Residual_Learning_for_Depth_Map_Super-Resolution_CVPR_2020_paper.pdf">Channel Attention based Iterative Residual Kearning for Depth Map Super-Resolution</a> 

## Stereo Image Super-Resolution

The dual camera has been widely used to estimate depth information. Meanwhile, stereo imaging can also be applied in image restoration. In the stereo image pair, we have two images with disparity much larger than one pixel. Therefore, full use of these two images can enhance the spatial resolution. 

[1] <a href="https://openaccess.thecvf.com/content_cvpr_2018/papers/Jeon_Enhancing_the_Spatial_CVPR_2018_paper.pdf">Enhancing the Spatial Resolution of Stereo Images Using A Parallax Prior</a> 

[2] <a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Learning_Parallax_Attention_for_Stereo_Image_Super-Resolution_CVPR_2019_paper.pdf">Learning Parallax Attention for Stereo Image Super-Resolution</a> 

[3] <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9206116">Parallax Attention for Unsupervised Stereo Correspondence Learning</a> 

[4] <a href="https://openaccess.thecvf.com/content_ICCVW_2019/papers/LCI/Wang_Flickr1024_A_Large-Scale_Dataset_for_Stereo_Image_Super-Resolution_ICCVW_2019_paper.pdf">Flickr1024: A Large-Scale Dataset for Stereo Image Super-Resolution</a> 

[5] <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8998204">A Stereo Attention Module for Stereo Image Super-Resolution</a> 

[6] <a href="https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Wang_Symmetric_Parallax_Attention_for_Stereo_Image_Super-Resolution_CVPRW_2021_paper.pdf">Symmetric Parallax Attention for Stereo Image Super-Resolution</a> 

[7] <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9382858">Deep Bilateral Learning for Stereo Image Super-Resolution</a>  

[8] <a href="https://ojs.aaai.org//index.php/AAAI/article/view/6880">Stereoscopic Image Super-Resolution with Stereo Consistent Feature</a> 

[9] <a href="https://arxiv.org/pdf/2106.00985.pdf">Feedback Network for Mutually Boosted Stereo Image Super-Resolution and Disparity Estimation</a> 

## Video Super-Resolution

As an emerging medium, video has attracted increasing attention owing to its ability to carry more information. Specifically, the video consists of multiple images, and each frame is an image, so it can provide more scene information. However, it is difficult to obtain high-resolution video due to the limitations of the network transmission and device storage. Therefore, video super-resolution (VSR) technology is essential. For VSR, multiple frames provide much more scene information, thus full use of the inter-frame temporal dependency (e.g., motions, brightness, color changes) is beneficial for high-quality video reconstruction. 

[1] <a href="https://openaccess.thecvf.com/content_cvpr_2017/papers/Caballero_Real-Time_Video_Super-Resolution_CVPR_2017_paper.pdf">Real-Time Video Super-Resolution with Spatio-Temporal Networks and Motion Compensation</a> 

[2] <a href="https://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Robust_Video_Super-Resolution_ICCV_2017_paper.pdf">Robust Video Super-Resolution with Learned Temporal Dynamics</a> 

[3] <a href="https://openaccess.thecvf.com/content_CVPR_2020/papers/Tian_TDAN_Temporally-Deformable_Alignment_Network_for_Video_Super-Resolution_CVPR_2020_paper.pdf">TDAN: Temporally-Deformable Alignment Network for Video Super-Resolution</a> 

[4] <a href="https://link.springer.com/chapter/10.1007/978-3-030-58610-2_38">Video Super-Resolution with Recurrent Structure-Detail Network</a> 

[5] <a href="https://arxiv.org/abs/2106.06847">Video Super-Resolution Transformer</a> 

# RECONSTRUCTION RESULTS

PSNR/SSIM comparison of lightweight SISR models (the number of model parameters less than 1000K) on Set5 (x4), Set14 (x4), and Urban100 (x4). Meanwhile, the training datasets and the number of model parameters are provided. Sort by PSNR of Set5 in ascending order. Best results are highlighted.

<p align="center">
<img src="Images/Lightweight-Results.png" width="1000px"/>
</p>

PSNR/SSIM comparison of large SISR models (the number of model parameters more than 1M, M=million) on Set5 (x4), Set14 (x4), and Urban100 (x4). Meanwhile, the training datasets and the number of model parameters are provided. Sort by PSNR of Set5 in ascending order. Best results are highlighted.

<p align="center">
<img src="Images/Large-Results.png" width="1000px"/>
</p>


```
@article{li2021beginner,
  title={From Beginner to Master: A Survey for Deep Learning-based Single-Image Super-Resolution},
  author={Li, Juncheng and Pei, Zehua and Zeng, Tieyong},
  journal={arXiv preprint arXiv:2109.14335},
  year={2021}
}
```

