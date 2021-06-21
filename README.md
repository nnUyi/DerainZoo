# DerainZoo (Single Image vs. Video Based)
[Youzhao Yang](https://github.com/nnuyi), [Hong Lu](http://homepage.fudan.edu.cn/honglu/machine-vision-lab/) in [Fudan Machine Vision Lab](https://github.com/FudanMV)

## 1 Description
   * DerainZoo: A list of deraining methods. Papers, codes and datasets are maintained. Other sources about deraining can be observed in [web1](https://github.com/TaiXiangJiang/FastDeRain) and [web2](https://github.com/hongwang01/Video-and-Single-Image-Deraining).

   * Datasets for single image deraining are available at the [website](https://github.com/nnUyi/DerainZoo/blob/master/DerainDatasets.md).
   
   * More datasets about other image processing task (brightening, HDR, color enhancement, and inpainting) are available [here](https://github.com/nnUyi/Image-Processing-Datasets).

## 2 Image Quality Metrics
* PSNR (Peak Signal-to-Noise Ratio) [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4550695) [[matlab code]](https://www.mathworks.com/help/images/ref/psnr.html) [[python code]](https://github.com/aizvorski/video-quality)
* SSIM (Structural Similarity) [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1284395) [[matlab code]](http://www.cns.nyu.edu/~lcv/ssim/ssim_index.m) [[python code]](https://github.com/aizvorski/video-quality/blob/master/ssim.py)
* VIF (Visual Quality) [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1576816) [[matlab code]](http://sse.tongji.edu.cn/linzhang/IQA/Evalution_VIF/eva-VIF.htm)
* FSIM (Feature Similarity) [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5705575) [[matlab code]](http://sse.tongji.edu.cn/linzhang/IQA/FSIM/FSIM.htm)
* NIQE (Naturalness Image Quality Evaluator) [[paper]](http://live.ece.utexas.edu/research/Quality/niqe_spl.pdf)[[matlab code]](http://live.ece.utexas.edu/research/Quality/index_algorithms.htm)[[python code]](https://github.com/aizvorski/video-quality/blob/master/niqe.py)

**Image & Video Quality Assessment Algorithms [[software release]](http://live.ece.utexas.edu/research/Quality/index_algorithms.htm)[[Texas Lab]](http://live.ece.utexas.edu/research/quality/)**

## 3 Single Image Deraining
### 3.1 Datasets
------------
#### 3.1.1 Synthetic Datasets
* Rain12 [[paper](https://ieeexplore.ieee.org/document/7780668/)] [[dataset](http://yu-li.github.io/paper/li_cvpr16_rain.zip)] (2016 CVPR)
* Rain100L_old_version [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_Deep_Joint_Rain_CVPR_2017_paper.pdf)][[dataset](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)](2017 CVPR)
  * Rain100L_new_version [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_Deep_Joint_Rain_CVPR_2017_paper.pdf)][[dataset](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)]
* Rain100H_old_version [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_Deep_Joint_Rain_CVPR_2017_paper.pdf)][[dataset](https://github.com/nnUyi/DerainZoo/blob/master/DerainDatasets.md)](2017 CVPR)
  * Rain100H_new_version [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_Deep_Joint_Rain_CVPR_2017_paper.pdf)][[dataset](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)]
* Rain800 [[paper](https://arxiv.org/abs/1701.05957)][[dataset](https://github.com/hezhangsprinter/ID-CGAN)] (2017 Arxiv)
* Rain1200 [[paper](https://arxiv.org/abs/1802.07412)][[dataset](https://github.com/hezhangsprinter/DID-MDN)] (2018 CVPR)
* Rain1400 [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Removing_Rain_From_CVPR_2017_paper.pdf)][[dataset](https://xueyangfu.github.io/projects/cvpr2017.html)] (2017 CVPR)
* Heavy Rain Dataset [[paper](http://export.arxiv.org/pdf/1904.05050)][[dataset](https://drive.google.com/file/d/1rFpW_coyxidYLK8vrcfViJLDd-BcSn4B/view)] (2019 CVPR)

#### 3.1.2 Real-world Datasets
* Practical_by_Yang [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_Deep_Joint_Rain_CVPR_2017_paper.pdf)][[dataset](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)] (2017 CVPR)
* Practica_by_Zhang [[paper](https://arxiv.org/abs/1701.05957)][[dataset](https://github.com/hezhangsprinter/ID-CGAN)] (2017 Arxiv)
* Real-world Paired Rain Dataset [[paper](https://arxiv.org/pdf/1904.01538.pdf)][[dataset](https://stevewongv.github.io/derain-project.html)] (2019 CVPR)

### 3.2 Papers
--------------
### 2021
* IDCL [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhou_Image_De-Raining_via_Continual_Learning_CVPR_2021_paper.pdf)][[code]()][[web]()]
   * Zhou, Man etc. Image De-raining via Continual Learning. (CVPR 2021)

* DRG [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yue_Semi-Supervised_Video_Deraining_With_Dynamical_Rain_Generator_CVPR_2021_paper.pdf)][[code](https://github.com/zsyOAOA/S2VD)][[web](https://github.com/zsyOAOA)]
   * Yue, Zongsheng etc. Semi-Supervised Video Deraining with Dynamical Rain Generator. (CVPR 2021)

* MPRNet [[paper](https://arxiv.org/abs/2102.02808)][[code](https://github.com/swz30/MPRNet)][[web](https://github.com/swz30)]
   * Zamir et al. Multi-Stage Progressive Image Restoration. (CVPR 2021)

* ADN [paper][[code](https://github.com/nnUyi/ADN)][[web](https://github.com/nnUyi)]
   * Yang, Youzhao etc. A Fast And Eefficient Network for Single Image Deraining. (ICASSP 2021)

* DualGCN [[paper](https://xueyangfu.github.io/paper/2021/AAAI/Preprint.pdf)][[code](https://xueyangfu.github.io/paper/2021/AAAI/code.zip)][[web](https://xueyangfu.github.io)]
   * Fu, Xueyang etc. Rain Streak Removal via Dual Graph Convolutional Network. (AAAI 2021)
   
* IPT [[paper](https://arxiv.org/pdf/2012.00364.pdf)][[code]()][[web]()]
   * Chen, Hanting etc. Pre-Trained Image Processing Transformer. (Arxiv 2020)

### 2020
* WDNet [[paper](https://arxiv.org/pdf/2008.00823.pdf)][[code]()][[web]()]
   * Liu, Lin etc. Wavelet-Based Dual-Branch Network for Image Demoir´eing. (2020 ECCV)

* Rethinking Image Deraining [[paper](https://arxiv.org/pdf/2008.00823.pdf)][[code](https://github.com/yluestc/derain)][[web](https://github.com/yluestc)]
   * Wang, Yinglong etc. Rethinking Image Deraining via Rain Streaks and Vapors. (2020 ECCV)

* JDNet [[paper](https://arxiv.org/pdf/2008.02763.pdf)][[code](https://github.com/Ohraincu/JDNet)][[web](https://github.com/Ohraincu)]
   * Wang, Cong etc. Joint Self-Attention and Scale-Aggregation for Self-Calibrated Deraining Network. (2020 ACMMM)

* DCSFN [[paper](https://arxiv.org/pdf/2008.00767.pdf)][[code]( https://github.com/Ohraincu/DCSFN)][[web](https://github.com/Ohraincu)]
   * Wang, Cong etc. DCSFN: Deep Cross-scale Fusion Network for Single Image Rain Removal. (2020 ACMMM)

* CVID [[paper](https://arxiv.org/pdf/2004.11373.pdf)][[code](https://github.com/Yingjun-Du/VID)][[web](https://github.com/Yingjun-Du)]
   * Du, Yingjun etc. Conditional Variational Image Deraining. (2020 TIP)

* DRD-Net [[paper]()][[web](https://github.com/Dengsgithub)][[code](https://github.com/Dengsgithub/DRD-Net)]
  * Deng, Sen etc. Detail-recovery Image Deraining via Context Aggregation Networks. (2020 CVPR)

* RCDNet [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_A_Model-Driven_Deep_Neural_Network_for_Single_Image_Rain_Removal_CVPR_2020_paper.pdf)][[web]()][[code](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Wang_A_Model-Driven_Deep_CVPR_2020_supplemental.pdf)]
  * Wang, Hong etc. A Model-driven Deep Neural Network for Single Image Rain Removal. (2020 CVPR)

* Syn2Rel [[paper](https://arxiv.org/pdf/2006.05580.pdf)][[web](https://github.com/rajeevyasarla)][[code](https://github.com/rajeevyasarla/Syn2Real)]
   * Rajeev Yasarla et al. Syn2Real Transfer Learning for Image Deraining using Gaussian Processes. (2020 CVPR)

* MSPFN [[paper](https://arxiv.org/pdf/2003.10985.pdf)][[code](https://github.com/kuihua/MSPFN)][[web](https://github.com/kuihua)]
   * Jiang Kui et al. Multi-Scale Progressive Fusion Network for Single Image Deraining. (2020 CVPR)

* Physical Model Guided ID [[paper](https://arxiv.org/pdf/2003.13242.pdf)][[code](https://supercong94.wixsite.com/supercong94)][[web](https://supercong94.wixsite.com/supercong94)]
   * Cong Wang et al. Physical Model Guided Deep Image Deraining. (2020 ICME)

* RDDAN [[paper](https://ieeexplore.ieee.org/abstract/document/9102945/)][[code](https://github.com/nnUyi/RDDAN)][[website](https://github.com/nnUyi)]
   * Yang, Youzhao et al. RDDAN: A Residual Dense Dilated Aggregated Network for Single Image Deraining. (2020 ICME)
   
* DiG-CoM [[paper](https://www.computer.org/csdl/proceedings-article/icme/2020/09102800/1kwqO6toxQk)][[code](https://github.com/nnUyi/DiG-CoM)][[website](https://github.com/nnUyi)]
   * Ran, Wu; Yang, Youzhao et al. Single Image Rain Removal Boosting via Directional Gradient. (2020 ICME)

* VID [[paper](http://openaccess.thecvf.com/content_WACV_2020/papers/Du_Variational_Image_Deraining_WACV_2020_paper.pdf)][code][[web](https://csjunxu.github.io/)]
   * Xu, Jun et al. Variational Image Deraining. (2020 WACV)

* CMGD [[paper](https://ieeexplore.ieee.org/abstract/document/9007569)][code][[web](https://github.com/rajeevyasarla)]
   * Rajeev Yasarla et al. Confidence Measure Guided Single Image De-Raining. (2020 TIP)

### 2019
* Survey [[paper](https://arxiv.org/pdf/1912.07150.pdf)][code][web]
   * Yang, Wenhan et al. Single Image Deraining: From Model-Based to Data-Driven and Beyond. (2019 TPAMI)
   
* RWL [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8610325)][code][web]
   * Yang, Wenhan et al. Scale-Free Single Image Deraining Via VisibilityEnhanced Recurrent Wavelet Learning. (2019 TIP)

* DPRDN [[paper](https://arxiv.org/pdf/1908.10521.pdf)][code][web]
   * Wei, Yanyan et al. A Coarse-to-Fine Multi-stream Hybrid Deraining Network for Single Image Deraining. (2019 ICDM)
   
* Survey [[paper](https://arxiv.org/pdf/1909.08326.pdf)][[code](https://github.com/hongwang01/Video-and-Single-Image-Deraining)][web]
   * Wang, Hong et al. A Survey on Rain Removal from Video and Single Image. (2019 Arxiv)

* ERL-Net [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_ERL-Net_Entangled_Representation_Learning_for_Single_Image_De-Raining_ICCV_2019_paper.pdf)][[code](https://github.com/RobinCSIRO/ERL-Net-for-Single-Image-Deraining)][web]
   * Wang, Guoqing et al. ERL-Net: Entangled Representation Learning for Single Image De-Raining. (2019 ICCV)

* ReHEN [[paper](http://delivery.acm.org/10.1145/3360000/3351149/p1814-yang.pdf?ip=202.120.235.180&id=3351149&acc=OPEN&key=BF85BBA5741FDC6E%2E88014DC677A1F2C3%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1573634982_715c64cb335fa08b82d82225f1944231#URLTOKEN#)][[code](https://github.com/nnUyi/ReHEN)][[web](https://nnuyi.github.io/)]
   * Yang, Youzhao et al. Single Image Deraining via Recurrent Hierarchy and Enhancement Network. (2019 ACM'MM)

* DTDN [[paper](http://delivery.acm.org/10.1145/3360000/3350945/p1833-wang.pdf?ip=202.120.235.223&id=3350945&acc=OPEN&key=BF85BBA5741FDC6E%2E88014DC677A1F2C3%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1572964912_ad2b0e3c2bc1fdb6f216a99468d1a0ea#URLTOKEN#)][code][web]
   * Wang, Zheng et al. DTDN: Dual-task De-raining Network. (2019 ACM'MM)
   
* GraNet [[paper](http://delivery.acm.org/10.1145/3360000/3350883/p1795-yu.pdf?ip=202.120.235.223&id=3350883&acc=OPEN&key=BF85BBA5741FDC6E%2E88014DC677A1F2C3%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1572964981_badf5608c2c0c67afa35ba86f50fe968#URLTOKEN#)][code][web]
   * Yu, Weijiang et al. Gradual Network for Single Image De-raining. (2019 ACM'MM)

* AMPE-Net [[paper](https://arxiv.org/pdf/1905.05404.pdf)][code][web]
   * Wang, Yinglong et al. An Effective Two-Branch Model-Based Deep Network for Single Image Deraining. (2019 Arxiv)

* ReMAEN [[paper](https://github.com/nnUyi/ReMAEN/tree/master/paper)][[code](https://github.com/nnUyi/ReMAEN)][[web](https://nnuyi.github.io/)]
   * Yang, Youzhao el al. Single Image Deraining using a Recurrent Multi-scale Aggregation and Enhancement Network. (2019 ICME)

* Rain Wiper [[paper](https://share.weiyun.com/5MXcnlX)][code][web]
   * Liang, Xiwen et al. Rain Wiper: An Incremental Randomly Wired Network for Single Image Deraining. (2019 PG)

* Dual-ResNet [[paper](https://arxiv.org/pdf/1903.08817v1.pdf)][[code](https://github.com/liu-vis/DualResidualNetworks)][web]
   * Liu, Xing et al. Dual Residual Networks Leveraging the Potential of Paired Operations for Image Restoration. (2019 CVPR)

* Heavy Rain Image Restoration [[paper](http://export.arxiv.org/pdf/1904.05050)][[code](https://github.com/liruoteng/HeavyRainRemoval)][[dataset](https://drive.google.com/file/d/1rFpW_coyxidYLK8vrcfViJLDd-BcSn4B/view)][web]
  * Li, Ruoteng et al. Heavy Rain Image Restoration: Integrating Physics Model and Conditional Adversarial Learning. (2019 CVPR)

* SPANet [[paper](https://arxiv.org/pdf/1904.01538.pdf)][[code](https://github.com/stevewongv/SPANet)][[web](https://stevewongv.github.io/derain-project.html)][[dataset](https://stevewongv.github.io/derain-project.html)]
  * Wang, Tianyu et al. Spatial Attentive Single-Image Deraining with a High Quality Real Rain Dataset. (2019 CVPR)

* Comprehensive Benchmark Analysis [[paper](https://arxiv.org/pdf/1903.08558.pdf)][[code](https://github.com/lsy17096535/Single-Image-Deraining)][[dataset](https://github.com/lsy17096535/Single-Image-Deraining)]
   * Li, Siyuan et al. Single Image Deraining: A Comprehensive Benchmark Analysis. (2019 CVPR)

* DAF-Net [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Hu_Depth-Attentional_Features_for_Single-Image_Rain_Removal_CVPR_2019_paper.pdf)][[code](https://github.com/xw-hu/DAF-Net)][[web](https://xw-hu.github.io/)]
   * Hu, Xiaowei et al. Depth-attentional Features for Single-image Rain Removal. (2019 CVPR)

* Semi-supervised Transfer Learning [[paper](https://arxiv.org/pdf/1807.11078.pdf)][[code](https://github.com/wwzjer/Semi-supervised-IRR)][web]
   * Wei, Wei et al. Semi-supervised Transfer Learning for Image Rain Removal. (2019 CVPR)

* PReNet [[paper](https://arxiv.org/pdf/1901.09221.pdf)][[code](https://github.com/csdwren/PReNet)][web]
   * Ren, Dongwei et al. Progressive Image Deraining Networks: A Better and Simpler Baseline. (2019 CVPR)

* UMRL-using-Cycle-Spinning [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yasarla_Uncertainty_Guided_Multi-Scale_Residual_Learning-Using_a_Cycle_Spinning_CNN_for_CVPR_2019_paper.pdf)][[code](https://github.com/rajeevyasarla/UMRL--using-Cycle-Spinning)][[web](https://github.com/rajeevyasarla)]
   * Rajeev Yasarla et al. Uncertainty Guided Multi-Scale Residual Learning-using a Cycle Spinning CNN for Single Image De-Raining. (2019 CVPR)

* RR-GAN [[paper](http://vijaychan.github.io/Publications/2019_derain.pdf)][code][web]
   * Zhu, Hongyuan et al. RR-GAN: Single Image Rain Removal Without Paired Information. (2019 AAAI)

* LPNet [[paper](https://arxiv.org/abs/1805.06173)][[code](https://xueyangfu.github.io/projects/LPNet.html)][[web](https://xueyangfu.github.io/)]
   * Fu, Xueyang et al. Lightweight Pyramid Networks for Image Deraining. (2019 TNNLS)

* Morphological Networks [[paper](https://arxiv.org/pdf/1901.02411.pdf)][code][web]
   * Mondal et al. Morphological Networks for Image De-raining. (2019 Arxiv)

### 2018

* QS Priors [[paper](https://arxiv.org/pdf/1812.08348.pdf)][code][web]
   * Wang et al. Rain Removal By Image Quasi-Sparsity Priors. (2018 Arxiv)

* Linear model [[paper](https://arxiv.org/pdf/1812.07870.pdf)][code][web]
   * Wang et al. Removing rain streaks by a linear model. (2018 Arxiv)

* Kernel Guided CNN [[paper](https://arxiv.org/pdf/1808.08545.pdf)][code][web]
   * Deng et al. Rain Streak Removal for Single Image via Kernel Guided CNN. (2018 Arxiv)

* Physics-Based GAM [[paper](https://arxiv.org/pdf/1808.00605.pdf)][[code](https://sites.google.com/site/jspanhomepage/physicsgan/)][web]
   * Pan, Jinshan et al. Physics-Based Generative Adversarial Models for Image Restoration and Beyond. (2018 Arxiv)
   
* Self-supervised Constraints [[paper](https://arxiv.org/pdf/1811.08575.pdf)][code][paper]
   * Jin et al. Unsupervised Single Image Deraining with Self-supervised Constraints. (2018 Arxiv)

* SRSE-Net [[paper](https://arxiv.org/pdf/1811.04761.pdf)][code][web]
   * Ye et al. Self-Refining Deep Symmetry Enhanced Network for Rain Removal. (2018 Arxiv)
  
* Tree-Structured Fusion Model [[paper](https://arxiv.org/pdf/1811.08632.pdf)][code][web]
   * Fu, Xueyang et. al. A Deep Tree-Structured Fusion Model for Single Image Deraining. (2018 Arxiv)

* Deep DCNet [[paper](https://arxiv.org/abs/1804.02688)][code]
 [[web1](https://sites.google.com/view/xjguo/homepage)] [[web2](https://sites.google.com/view/xjguo/homepage)]
   * Li, Siyuan et al. Fast Single Image Rain Removal via a Deep Decomposition-Composition Network. (ArXiv2018)
 
*  SFARL Model [[paper](https://arxiv.org/abs/1804.04522)][code][[web](https://sites.google.com/site/csrendw/home)]
   * Ren, Dongwei et al. Simultaneous Fidelity and Regularization Learning for Image Restoration. (ArXiv2018)

* GCAN [[paper](https://arxiv.org/pdf/1811.08747.pdf)][[code](https://github.com/cddlyf/GCANet)][web]
   * Chen et. al. Gated Context Aggregation Network for Image Dehazing and Deraining. (2018 WACV)
  
* Cycle-GAN [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8397790)][code][web]
   * Pu, Jinchuan et al. Removing rain based on a Cycle Generative Adversarial Network. (2018 ICIEA)

*  RESCAN [[paper](https://arxiv.org/pdf/1807.05698.pdf)][[code](https://xialipku.github.io/RESCAN/)][[web](https://xialipku.github.io/RESCAN/)]
   * Li, Xia et al. Recurrent Squeeze-and-Excitation Context Aggregation Net for Single Image Deraining. (2018 ECCV)

* RGFFN [[paper](https://arxiv.org/abs/1804.07493)][code][web]
   * Fan, Zhiwen et al. Residual-Guide Feature Fusion Network for Single Image Deraining. (2018 ACM'MM)

* NLEDN [[paper](https://arxiv.org/pdf/1808.01491.pdf)][[code](https://github.com/AlexHex7/NLEDN)][web]
   * Li, Guanbin et al. Non-locally Enhanced Encoder-Decoder Network for Single Image De-raining. (2018 ACM'MM)
    
* DualCNN [[paper](http://faculty.ucmerced.edu/mhyang/papers/cvpr2018_dual_cnn.pdf)][[code](https://sites.google.com/site/jspanhomepage/dualcnn)][[web](https://sites.google.com/site/jspanhomepage/dualcnn)]
  * Pan, Jinshan et al. Learning Dual Convolutional Neural Networks for Low-Level Vision. (2018 CVPR)
  
* Attentive GAN [[paper](https://arxiv.org/abs/1711.10098)][[code](https://github.com/rui1996/DeRaindrop)][[web](https://rui1996.github.io/)][[project](https://rui1996.github.io/raindrop/raindrop_removal.html)] [[reimplement code](https://github.com/MaybeShewill-CV/attentive-gan-derainnet)]
    * Qian, Rui et al. Attentive Generative Adversarial Network for Raindrop Removal from a Single Image. (2018 CVPR)
  (*tips: this research focuses on reducing the effets form the adherent rain drops instead of rain streaks removal*)

* DID-MDN [[paper](https://arxiv.org/abs/1802.07412)][[code](https://github.com/hezhangsprinter/DID-MDN)][[web](https://sites.google.com/site/hezhangsprinter/)] 
  * Zhang, He et al. Density-aware Single Image De-raining using a Multi-stream Dense Network. (2018 CVPR)
  
* Directional global sparse model [[paper](https://www.sciencedirect.com/science/article/pii/S0307904X18301069)]
 [[code](http://www.escience.cn/system/file?fileId=98760)][[web](http://www.escience.cn/people/dengliangjian/index.html)]
  * Deng, Liangjian et al. A directional global sparse model for single image rain removal. (2018 AMM)

* Gradient domain [[paper](https://www.sciencedirect.com/science/article/pii/S0031320318300700)][code][web]
  * Du, Shuangli et al. Single image deraining via decorrelating the rain streaks and background scene in gradient domain. (2018 PR)

### 2017
* ID_CGAN [[paper](https://arxiv.org/abs/1701.05957)][[code](https://github.com/hezhangsprinter/ID-CGAN)] [[web](http://www.rci.rutgers.edu/~vmp93/index_ImageDeRaining.html)]
  * Zhang, He et al. Image De-raining Using a Conditional Generative Adversarial Network. (2017 Arxiv)

* Transformed Low-Rank Model [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Chang_Transformed_Low-Rank_Model_ICCV_2017_paper.html)][code][web]
  * Chang, Yi et al. Transformed Low-Rank Model for Line Pattern Noise Removal. (2017 ICCV)

* JBO [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Joint_Bi-Layer_Optimization_ICCV_2017_paper.html)][code][[web](http://appsrv.cse.cuhk.edu.hk/~lzhu/)] 
  * Wei, Wei et al. Joint Bi-layer Optimization for Single-image Rain Streak Removal. (2017 ICCV)

* JCAS [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Gu_Joint_Convolutional_Analysis_ICCV_2017_paper.html)][[code](http://www4.comp.polyu.edu.hk/~cslzhang/code/JCAS_Release.zip)][[web](https://sites.google.com/site/shuhanggu/home)]
  * Gu, Shuhang et al. Joint Convolutional Analysis and Synthesis Sparse Representation for Single Image Layer Separation. (2017 ICCV)

* DDN [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Removing_Rain_From_CVPR_2017_paper.pdf)] [[code](https://xueyangfu.github.io/projects/cvpr2017.html)][[web](https://xueyangfu.github.io/projects/cvpr2017.html)]
  * Fu, Xueyang et al. Removing rain from single images via a deep detail network. (2017 CVPR)
  
* JORDER [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_Deep_Joint_Rain_CVPR_2017_paper.pdf)] [[code](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)][[web](http://www.icst.pku.edu.cn/struct/people/whyang.html)]
  * Yang, Wenhan et al. Deep joint rain detection and removal from a single image. (2017 CVPR)
 
* Hierarchical Approach [[paper](http://ieeexplore.ieee.org/abstract/document/7934435/)][code][web]
  * Wang, Yinglong et al. A Hierarchical Approach for Rain or Snow Removing in a Single Color Image. (2017 TIP)

* Clearing The Skies [[paper](https://ieeexplore.ieee.org/abstract/document/7893758/)][[code](https://xueyangfu.github.io/projects/tip2017.html)][[web](https://xueyangfu.github.io/projects/tip2017.html)]
  * Fu, Xueyang et al. Clearing the skies: A deep network architecture for single-image rain removal. (2017 TIP)

* Error-optimized Sparse Representation [[paper](https://ieeexplore.ieee.org/abstract/document/7878618/)][code][web]
  * Chen, Bohao et al. Error-optimized sparse representation for single image rain removal. (2017 TIE)

### 2015-2016
* LP(GMM) (2016 CVPR, 2017 TIP)
  * Li, Yu et al. Rain streak removal using layer priors. [[paper](https://ieeexplore.ieee.org/document/7780668/)][code][web]
  * Li, Yu et al. Single Image Rain Streak Decomposition Using Layer Priors. [[paper](https://ieeexplore.ieee.org/abstract/document/7934436/)]
 [[dataset](http://yu-li.github.io/paper/li_cvpr16_rain.zip)][[web](http://yu-li.github.io/)]

* DSC [[paper](http://ieeexplore.ieee.org/document/7410745/)][[code](http://www.math.nus.edu.sg/~matjh/download/image_deraining/rain_removal_v.1.1.zip)][web]
  * Luo, Yu et al. Removing rain from a single image via discriminative sparse coding. (2015 ICCV)

* Window Covered [[paper](https://cs.nyu.edu/~deigen/rain/)][[code](https://cs.nyu.edu/~deigen/rain/)][web]
  * David, Eigen et al. Restoring An Image Taken Through a Window Covered with Dirt or Rain. (2013 ICCV)

* Image Decomposition [paper](http://www.ee.nthu.edu.tw/cwlin/Rain_Removal/tip_rain_removal_2011.pdf)][[code](http://www.ee.nthu.edu.tw/~cwlin/pub.htm)][web]
  * Kang, Liwei et al. Automatic Single-Image-Based Rain Streaks Removal via Image Decomposition. (2012 TIP)
  
## 4 Video Based Deraining
### 2019
* D3R-Net [[paper](http://www.icst.pku.edu.cn/struct/Pub%20Files/2019/ywh_tip19.pdf)][code][web]
   * Yang, Wenhan et al. D3R-Net: Dynamic Routing Residue Recurrent Network for Video Rain Removal. (2019 TIP)

### 2018
* MSCSC [[paper](https://pan.baidu.com/s/1iiRr7ns8rD7sFmvRFcxcvw)][[code](https://github.com/MinghanLi/MS-CSC-Rain-Streak-Removal)] [[web](https://sites.google.com/view/cvpr-anonymity)][[video](https://www.youtube.com/watch?v=tYHX7q0yK4M)]
    * Li, Minghan et al. Video Rain Streak Removal By Multiscale ConvolutionalSparse Coding. (2018 CVPR)

* CNN Framework [[paper](https://arxiv.org/abs/1803.10433)][code][[web Chen](https://github.com/hotndy/SPAC-SupplementaryMaterials)] [[web Chau](http://www.ntu.edu.sg/home/elpchau/)]
  * Chen, Jie et al. Robust Video Content Alignment and Compensation for Rain Removal in a CNN Framework. (2018 CVPR)
  * Chen, Jie et al. Robust Video Content Alignment and Compensation for Clear Vision Through the Rain [[paper](https://arxiv.org/abs/1804.09555)][code][web](*tips: I guess this is the extended journal version*)

* Erase or Fill [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Erase_or_Fill_CVPR_2018_paper.pdf)][[code](https://github.com/flyywh/J4RNet-Deep-Video-Deraining-CVPR-2018)][[web Liu](http://www.icst.pku.edu.cn/struct/people/liujiaying.html)] [[web Yang](http://www.icst.pku.edu.cn/struct/people/whyang.html)]
    * Liu, Jiaying et al. Erase or Fill? Deep Joint Recurrent Rain Removal and Reconstruction in Videos. (2018 CVPR)

### 2017
* MoG [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Wei_Should_We_Encode_ICCV_2017_paper.html)] 
[[code](https://github.com/wwxjtu/RainRemoval_ICCV2017)][[web](https://github.com/wwxjtu/RainRemoval_ICCV2017)]
  * Wei, Wei et al. Should We Encode Rain Streaks in Video as Deterministic or Stochastic? (2017 ICCV)

* FastDeRain [[paper](http://openaccess.thecvf.com/content_cvpr_2017/html/Jiang_A_Novel_Tensor-Based_CVPR_2017_paper.html)][[code](https://github.com/TaiXiangJiang/FastDeRain)]
  * Jiang, Taixiang et al. A novel tensor-based video rain streaks removal approach via utilizing discriminatively intrinsic priors. (2017 CVPR)

* Matrix Decomposition [[paper](http://openaccess.thecvf.com/content_cvpr_2017/html/Ren_Video_Desnowing_and_CVPR_2017_paper.html)][code][web]
  * Ren, Weilong et al. Video Desnowing and Deraining Based on Matrix Decomposition. (2017 CVPR)

### 2015-2016
* Adherent Raindrop Modeling [[paper](https://ieeexplore.ieee.org/abstract/document/7299675/)][code][[web](http://www.cvl.iis.u-tokyo.ac.jp/~yousd/CVPR2013/Shaodi_CVPR2013.html)]
  * You, Shaodi et al. Adherent raindrop modeling, detectionand removal in video. (2016 TPAMI)

* Low-rank Matrix Completion [[paper](https://ieeexplore.ieee.org/abstract/document/7101234/)][[code](http://mcl.korea.ac.kr/~jhkim/deraining/)][web]
  * Kim, JH et al. Video deraining and desnowing using temporal correlation and low-rank matrix completion. (2015 TIP)

* Utilizing Local Phase Information [[paper](https://link.springer.com/article/10.1007/s11263-014-0759-8)][code][web]
  * Santhaseelan et al. Utilizing local phase information to remove rain from video. (2015 IJCV)

## 5 Acknowledgement
- Thanks for the sharing of codes of image quality metrics by [Wang, Hong](https://github.com/hongwang01/Video-and-Single-Image-Deraining).

## 6 Contact
- e-mail: yzyang17@fudan.edu.cn
