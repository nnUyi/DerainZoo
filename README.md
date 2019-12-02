|Name|E-mail|Group|
|---|---|---|
|Youzhao Yang|yzyang17@fudan.edu.cn|[Fudan Machine Vision Lab](https://github.com/FudanMV) by [Hong Lu](http://homepage.fudan.edu.cn/honglu/machine-vision-lab/)|

# DerainZoo (Single Image vs. Video Based)
   * DerainZoo: A list of deraining methods. Papers, codes and datasets are maintained. Other sources about deraining can be observed [here](https://github.com/TaiXiangJiang/FastDeRain).
   
   * Datasets for single image deraining are available at the [website](https://github.com/nnUyi/DerainZoo/blob/master/DerainDatasets.md).
   
   * More datasets about other image processing task (brightening, HDR, color enhancement, and inpainting) are available [here](https://github.com/nnUyi/Image-Processing-Datasets).

##  Image Quality Metrics
* PSNR (Peak Signal-to-Noise Ratio) [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4550695) [[matlab code]](https://www.mathworks.com/help/images/ref/psnr.html) [[python code]](https://github.com/aizvorski/video-quality)
* SSIM (Structural Similarity) [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1284395) [[matlab code]](http://www.cns.nyu.edu/~lcv/ssim/ssim_index.m) [[python code]](https://github.com/aizvorski/video-quality/blob/master/ssim.py)
* VIF (Visual Quality) [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1576816) [[matlab code]](http://sse.tongji.edu.cn/linzhang/IQA/Evalution_VIF/eva-VIF.htm)
* FSIM (Feature Similarity) [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5705575) [[matlab code]](http://sse.tongji.edu.cn/linzhang/IQA/FSIM/FSIM.htm))

- Please note that all quantitative results are computed based on Y channel. Thanks for the sharing of codes by [Hong Wang](https://github.com/hongwang01/Video-and-Single-Image-Deraining).

## Single Image Deraining
### Datasets
------------
#### Synthetic Datasets
* Rain12 [[paper](https://ieeexplore.ieee.org/document/7780668/)] [[dataset](http://yu-li.github.io/paper/li_cvpr16_rain.zip)] (2016 CVPR)
* Rain100L [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_Deep_Joint_Rain_CVPR_2017_paper.pdf)][[dataset](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)] (2017 CVPR)
* Rain100H [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_Deep_Joint_Rain_CVPR_2017_paper.pdf)][[dataset](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)] (2017 CVPR)
* Rain800 [[paper](https://arxiv.org/abs/1701.05957)][[dataset](https://github.com/hezhangsprinter/ID-CGAN)] (2017 Arxiv)
* Rain1200 [[paper](https://arxiv.org/abs/1802.07412)][[dataset](https://github.com/hezhangsprinter/DID-MDN)] (2018 CVPR)
* Rain1400 [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Removing_Rain_From_CVPR_2017_paper.pdf)][[dataset](https://xueyangfu.github.io/projects/cvpr2017.html)] (2017 CVPR)
* Heavy Rain Dataset [[paper](http://export.arxiv.org/pdf/1904.05050)][[dataset](https://drive.google.com/file/d/1rFpW_coyxidYLK8vrcfViJLDd-BcSn4B/view)] (2019 CVPR)

#### Real-world Datasets
* Practical_by_Yang [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_Deep_Joint_Rain_CVPR_2017_paper.pdf)][[dataset](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)] (2017 CVPR)
* Practica_by_Zhang [[paper](https://arxiv.org/abs/1701.05957)][[dataset](https://github.com/hezhangsprinter/ID-CGAN)] (2017 Arxiv)
* Real-world Paired Rain Dataset [[paper](https://arxiv.org/pdf/1904.01538.pdf)][[dataset](https://stevewongv.github.io/derain-project.html)] (2019 CVPR)

### Papers
--------------
2019
--
* A Survey on Rain Removal from Video and Single Image (2019 Arxiv)
   * Hong Wang et al. A Survey on Rain Removal from Video and Single Image. [[paper](https://arxiv.org/pdf/1909.08326.pdf)][[code](https://github.com/hongwang01/Video-and-Single-Image-Deraining)]

* ERL-Net: Entangled Representation Learning for Single Image De-Raining (2019 ICCV)
   * Guoqing Wang, Changming Sun et al. ERL-Net: Entangled Representation Learning for Single Image De-Raining [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_ERL-Net_Entangled_Representation_Learning_for_Single_Image_De-Raining_ICCV_2019_paper.pdf)][[codes](https://github.com/RobinCSIRO/ERL-Net-for-Single-Image-Deraining)]

* Single Image Deraining via Recurrent Hierarchy and Enhancement Network (2019 ACM MM)
   * Youzhao Yang et al. Single Image Deraining via Recurrent Hierarchy and Enhancement Network. [[paper](http://delivery.acm.org/10.1145/3360000/3351149/p1814-yang.pdf?ip=202.120.235.180&id=3351149&acc=OPEN&key=BF85BBA5741FDC6E%2E88014DC677A1F2C3%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1573634982_715c64cb335fa08b82d82225f1944231#URLTOKEN#)][[code](https://github.com/nnUyi/ReHEN)]

* DTDN: Dual-task De-raining Network (2019 ACM MM)
   * Wang Zheng et al. DTDN: Dual-task De-raining Network [[paper](http://delivery.acm.org/10.1145/3360000/3350945/p1833-wang.pdf?ip=202.120.235.223&id=3350945&acc=OPEN&key=BF85BBA5741FDC6E%2E88014DC677A1F2C3%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1572964912_ad2b0e3c2bc1fdb6f216a99468d1a0ea#URLTOKEN#)][[codes TAB]()]
   
* Gradual Network for Single Image De-raining (2019 ACM MM)
   * Yu Weijiang et al. Gradual Network for Single Image De-raining [[paper](http://delivery.acm.org/10.1145/3360000/3350883/p1795-yu.pdf?ip=202.120.235.223&id=3350883&acc=OPEN&key=BF85BBA5741FDC6E%2E88014DC677A1F2C3%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1572964981_badf5608c2c0c67afa35ba86f50fe968#URLTOKEN#)][[codes TAB]()]

* An Effective Two-Branch Model-Based Deep Network for Single Image Deraining (2019 Arxiv)
   * Wang Yinglong et al. An Effective Two-Branch Model-Based Deep Network for Single Image Deraining. [[paper](https://arxiv.org/pdf/1905.05404.pdf)]

* Single Image Deraining using a Recurrent Multi-scale Aggregation and Enhancement Network (2019 ICME)
   * Yang Youzhao el al. Single Image Deraining using a Recurrent Multi-scale Aggregation and Enhancement Network [[paper](https://github.com/nnUyi/ReMAEN/tree/master/paper)][[code](https://github.com/nnUyi/ReMAEN)]

* Dual Residual Networks Leveraging the Potential of Paired Operations for Image Restoration (2019 CVPR)
   * Liu Xing et al. Dual Residual Networks Leveraging the Potential of Paired Operations for Image Restoration [[paper](https://arxiv.org/pdf/1903.08817v1.pdf)][[code](https://github.com/liu-vis/DualResidualNetworks)]

* Heavy Rain Image Restoration: Integrating Physics Model and Conditional Adversarial Learning (2019 CVPR)
  * Li Ruoteng et al. Heavy Rain Image Restoration: Integrating Physics Model and Conditional Adversarial Learning [[paper](http://export.arxiv.org/pdf/1904.05050)][[code](https://github.com/liruoteng/HeavyRainRemoval)][[dataset](https://drive.google.com/file/d/1rFpW_coyxidYLK8vrcfViJLDd-BcSn4B/view)]

* Spatial Attentive Single-Image Deraining with a High Quality Real Rain Dataset (2019 CVPR)
  * Wang Tianyu et al. Spatial Attentive Single-Image Deraining with a High Quality Real Rain Dataset [[paper](https://arxiv.org/pdf/1904.01538.pdf)][[code](https://github.com/stevewongv/SPANet)][[github.io](https://stevewongv.github.io/derain-project.html)][[dataset](https://stevewongv.github.io/derain-project.html)]

* Single Image Deraining: A Comprehensive Benchmark Analysis (2019 CVPR)
   * Li Siyuan et al. Single Image Deraining: A Comprehensive Benchmark Analysis [[paper](https://arxiv.org/pdf/1903.08558.pdf)][[code](https://github.com/lsy17096535/Single-Image-Deraining)][[dataset TAB](https://github.com/lsy17096535/Single-Image-Deraining)]

* DAF-Net: Depth-attentional Features for Single-image Rain Removal (2019 CVPR)
   * Hu Xiaowei et al. Depth-attentional Features for Single-image Rain Removal [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Hu_Depth-Attentional_Features_for_Single-Image_Rain_Removal_CVPR_2019_paper.pdf)][[code](https://github.com/xw-hu/DAF-Net)][[github.io](https://xw-hu.github.io/)]

*  Semi-supervised Transfer Learning for Image Rain Removal (2019 CVPR)
   * Wei Wei et al. Semi-supervised Transfer Learning for Image Rain Removal [[paper](https://arxiv.org/pdf/1807.11078.pdf)][[code](https://github.com/wwzjer/Semi-supervised-IRR)]

*  PReNet: Progressive Image Deraining Networks: A Better and Simpler Baseline (2019 CVPR)
   * Ren Dongwei et al. Progressive Image Deraining Networks: A Better and Simpler Baseline [[paper](https://arxiv.org/pdf/1901.09221.pdf)][[code](https://github.com/csdwren/PReNet)]

*  RR-GAN: Single Image Rain Removal Without Paired Information (2019 AAAI)
   *  Zhu et al. RR-GAN: Single Image Rain Removal Without Paired Information [[paper](http://vijaychan.github.io/Publications/2019_derain.pdf)]

*  Lightweight Pyramid Networks for Image Deraining (2019 TNNLS)
   * Fu Xueyang et al. Lightweight Pyramid Networks for Image Deraining. [[paper](https://arxiv.org/abs/1805.06173)]  [[Dr. Xueyang Fu's homepage](https://xueyangfu.github.io/)][[code](https://xueyangfu.github.io/projects/LPNet.html)]

* Morphological Networks for Image De-raining (Arxiv2019)
   * Mondal et al. Morphological Networks for Image De-raining [[paper](https://arxiv.org/pdf/1901.02411.pdf)]

2018
--

* Rain Removal By Image Quasi-Sparsity Priors (Arxiv2018)
   * Wang et. al. Rain Removal By Image Quasi-Sparsity Priors [[paper](https://arxiv.org/pdf/1812.08348.pdf)]

* Removing rain streaks by a linear model (Arxiv2018)
   * Wang et. al. Removing rain streaks by a linear model [[paper](https://arxiv.org/pdf/1812.07870.pdf)]

* Rain Streak Removal for Single Image via Kernel Guided CNN (Arxiv2018)
   * Deng et. al. Rain Streak Removal for Single Image via Kernel Guided CNN [[paper](https://arxiv.org/pdf/1808.08545.pdf)]

* Physics-Based Generative Adversarial Models for Image Restoration and Beyond (Arxiv2018)
   * Pan Jinshan et al. Physics-Based Generative Adversarial Models for Image Restoration and Beyond [[paper](https://arxiv.org/pdf/1808.00605.pdf)][[code](https://sites.google.com/site/jspanhomepage/physicsgan/)]
   
* Unsupervised Single Image Deraining with Self-supervised Constraints (Arxiv2018)
  * Jin et. al. Unsupervised Single Image Deraining with Self-supervised Constraints [[paper](https://arxiv.org/pdf/1811.08575.pdf)]

* Self-Refining Deep Symmetry Enhanced Network for Rain Removal (Arxiv2018)
  * Ye et. al. Self-Refining Deep Symmetry Enhanced Network for Rain Removal [[paper](https://arxiv.org/pdf/1811.04761.pdf)]
  
* A Deep Tree-Structured Fusion Model for Single Image Deraining (Arxiv2018)
  * Fu Xueyang et. al. A Deep Tree-Structured Fusion Model for Single Image Deraining [[paper](https://arxiv.org/pdf/1811.08632.pdf)]

* Fast Single Image Rain Removal via a Deep Decomposition-Composition Network (ArXiv2018)
  * Li Siyuan et al. Fast Single Image Rain Removal via a Deep Decomposition-Composition Network [[paper](https://arxiv.org/abs/1804.02688)]
 [[Prof. Xiaojie Guo's homepage (code TBA)](https://sites.google.com/view/xjguo/homepage)] [[Prof. Wenqi Ren's homepage (code TBA)](https://sites.google.com/view/xjguo/homepage)]
 
* Simultaneous Fidelity and Regularization Learning for Image Restoration (ArXiv2018)
  * Ren Dongwei et al. Simultaneous Fidelity and Regularization Learning for Image Restoration. [[paper](https://arxiv.org/abs/1804.04522)]
 [[Ren's homepage](https://sites.google.com/site/csrendw/home)]

* Gated Context Aggregation Network for Image Dehazing and Deraining (2018 WACV)
  * Chen et. al. Gated Context Aggregation Network for Image Dehazing and Deraining [[paper](https://arxiv.org/pdf/1811.08747.pdf)][[code](https://github.com/cddlyf/GCANet)]
  
* Removing rain based on a Cycle Generative Adversarial Network (2018 ICIEA)
   * Pu Jinchuan et al. Removing rain based on a Cycle Generative Adversarial Network [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8397790)]

*  RESCAN: Recurrent Squeeze-and-Excitation Context Aggregation Net for Single Image Deraining (2018 ECCV)
   * Li Xia et al. Recurrent Squeeze-and-Excitation Context Aggregation Net for Single Image Deraining [[paper](https://arxiv.org/pdf/1807.05698.pdf)]  [[github.io](https://xialipku.github.io/RESCAN/)]

* Residual-Guide Feature Fusion Network for Single Image Deraining (2018 ACMMM)
  * Fan Zhiwen et al. Residual-Guide Feature Fusion Network for Single Image Deraining. [[paper](https://arxiv.org/abs/1804.07493)]

* NLEDN: Non-locally Enhanced Encoder-Decoder Network for Single Image De-raining (2018 ACMMM)
   * Li Guanbin et al. Non-locally Enhanced Encoder-Decoder Network for Single Image De-raining [[paper](https://arxiv.org/pdf/1808.01491.pdf)][[code](https://github.com/AlexHex7/NLEDN)]
    
* Dual CNN: Learning Dual Convolutional Neural Networks for Low-Level Vision (2018 CVPR)
  * Pan Jinshan et al. Learning Dual Convolutional Neural Networks for Low-Level Vision [[paper](http://faculty.ucmerced.edu/mhyang/papers/cvpr2018_dual_cnn.pdf)] [[project (trained model and codes available](https://sites.google.com/site/jspanhomepage/dualcnn)]
  
* Attentive GAN: Attentive Generative Adversarial Network for Raindrop Removal from a Single Image (2018 CVPR)
    * Qian Rui et al. Attentive Generative Adversarial Network for Raindrop Removal from a Single Image [[paper](https://arxiv.org/abs/1711.10098)]
[[Dr. Rui Qian's homepage](https://rui1996.github.io/)]  [[project](https://rui1996.github.io/raindrop/raindrop_removal.html)] [[reimplement code](https://github.com/MaybeShewill-CV/attentive-gan-derainnet)] [[code](https://github.com/rui1996/DeRaindrop)]
  (*tips: this research focuses on reducing the effets form the adherent rain drops instead of rain streaks removal*)

* DID-MDN: Density-aware Single Image De-raining using a Multi-stream Dense Network (2018 CVPR)
  * Zhang He et al. Density-aware Single Image De-raining using a Multi-stream Dense Network [[paper](https://arxiv.org/abs/1802.07412)] [[code](https://github.com/hezhangsprinter/DID-MDN)]  [[Dr. He Zhang's homepage](https://sites.google.com/site/hezhangsprinter/)] 
  
* A directional global sparse model for single image rain removal (2018 AMM)
  * Deng Liang Jian et al. A directional global sparse model for single image rain removal [[paper](https://www.sciencedirect.com/science/article/pii/S0307904X18301069)]
 [[code](http://www.escience.cn/system/file?fileId=98760)]  [[Dr. Deng's homepage](http://www.escience.cn/people/dengliangjian/index.html)]
(*tips: I am the last author, and your can also find the implemention in this repository*)

* Single image deraining via decorrelating the rain streaks and background scene in gradient domain (2018 PR)
  * Du Shuangli et al. Single image deraining via decorrelating the rain streaks and background scene in gradient domain [[paper](https://www.sciencedirect.com/science/article/pii/S0031320318300700)]

2017
--
* Conditional GAN: Image De-raining Using a Conditional Generative Adversarial Network (Arxiv2017)
  * Zhang He et al. Image De-raining Using a Conditional Generative Adversarial Network. [[paper](https://arxiv.org/abs/1701.05957)] [[code](https://github.com/hezhangsprinter/ID-CGAN)] [[Project](http://www.rci.rutgers.edu/~vmp93/index_ImageDeRaining.html)]

* Transformed Low-Rank Model for Line Pattern Noise Removal (2017 ICCV)
  * Chang Yi et al. Transformed Low-Rank Model for Line Pattern Noise Removal. [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Chang_Transformed_Low-Rank_Model_ICCV_2017_paper.html)]

* Joint Bi-layer Optimization for Single-image Rain Streak Removal (2017 ICCV)
  * Wei Wei et al. Joint Bi-layer Optimization for Single-image Rain Streak Removal. [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Joint_Bi-Layer_Optimization_ICCV_2017_paper.html)]
 [[Dr. Zhu's homepage](http://appsrv.cse.cuhk.edu.hk/~lzhu/)] 

* JCAS: Joint Convolutional Analysis and Synthesis Sparse Representation (2017 ICCV)
  * Gu Shuhang et al. Joint Convolutional Analysis and Synthesis Sparse Representation for Single Image Layer Separation. [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Gu_Joint_Convolutional_Analysis_ICCV_2017_paper.html)]
 [[code](http://www4.comp.polyu.edu.hk/~cslzhang/code/JCAS_Release.zip)]  [[Gu's homepage](https://sites.google.com/site/shuhanggu/home)]

* DDN: Removing rain from single images via a deep detail network (2017 CVPR)
  * Fu Xueyang et al. Removing rain from single images via a deep detail network. [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Removing_Rain_From_CVPR_2017_paper.pdf)] [[code](https://xueyangfu.github.io/projects/cvpr2017.html)]
  
* JODER: Deep joint rain detection and removal from a single image (2017 CVPR)
  * Yang Wenhan et al. Deep joint rain detection and removal from a single image. [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_Deep_Joint_Rain_CVPR_2017_paper.pdf)] [[code)](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)] [[Dr. Wenhan Yang's homepage](http://www.icst.pku.edu.cn/struct/people/whyang.html)]
 
* Hierarchical: A Hierarchical Approach for Rain or Snow Removing in a Single Color Image (2017 TIP)
  * Wang Yinglong et al. A Hierarchical Approach for Rain or Snow Removing in a Single Color Image. [[paper](http://ieeexplore.ieee.org/abstract/document/7934435/)]

* Clearing the skies: A deep network architecture for single-image rain removal (2017 TIP)
  * Fu Xue yang et al. Clearing the skies: A deep network architecture for single-image rain removal. [[paper](https://ieeexplore.ieee.org/abstract/document/7893758/)] [[code](https://xueyangfu.github.io/projects/tip2017.html)] 

* Error-optimized sparse representation for single image rain removal (2017 TIE)
  * Chen Bohao et al. Error-optimized sparse representation for single image rain removal. [[paper](https://ieeexplore.ieee.org/abstract/document/7878618/)]

2015-2016
--
* LP(GMM): Rain streak removal using layer priors (2016 CVPR, 2017 TIP)
  * Li Yu et al. Rain streak removal using layer priors. [[paper](https://ieeexplore.ieee.org/document/7780668/)]
  * Li Yu et al. Single Image Rain Streak Decomposition Using Layer Priors. [[paper](https://ieeexplore.ieee.org/abstract/document/7934436/)]
 [[rainy images dataset](http://yu-li.github.io/paper/li_cvpr16_rain.zip)] [[Li's homepage](http://yu-li.github.io/)]

* Discriminative sparse coding: Removing rain from a single image via discriminative sparse coding (2015 ICCV)
  * Luo Yu et al. Removing rain from a single image via discriminative sparse coding.  [[paper](http://ieeexplore.ieee.org/document/7410745/)] [[code](http://www.math.nus.edu.sg/~matjh/download/image_deraining/rain_removal_v.1.1.zip)]

* Restoring An Image Taken Through a Window Covered with Dirt or Rain (2013 ICCV)
  * David Eigen et al. Restoring An Image Taken Through a Window Covered with Dirt or Rain [[paper](https://cs.nyu.edu/~deigen/rain/)][[codes](https://cs.nyu.edu/~deigen/rain/)]

* Automatic Single-Image-Based Rain Streaks Removal via Image Decomposition (2012 TIP)
  * Kang Li Wei et al. Automatic Single-Image-Based Rain Streaks Removal via Image Decomposition [[paper](http://www.ee.nthu.edu.tw/cwlin/Rain_Removal/tip_rain_removal_2011.pdf)][[codes](http://www.ee.nthu.edu.tw/~cwlin/pub.htm)]
  
## Video Based Deraining
2019
--
* D3R-Net: Dynamic Routing Residue Recurrent Network for Video Rain Removal (2019 TIP)
   * Wenhan Yang et al. D3R-Net: Dynamic Routing Residue Recurrent Network for Video Rain Removal [[paper](http://www.icst.pku.edu.cn/struct/Pub%20Files/2019/ywh_tip19.pdf)][[code TAB]()]

2018
--
* MSCSC: Video Rain Streak Removal By Multiscale ConvolutionalSparse Coding (2018 CVPR)
    * Li Minghan et al. Video Rain Streak Removal By Multiscale ConvolutionalSparse Coding [[paper](https://pan.baidu.com/s/1iiRr7ns8rD7sFmvRFcxcvw)]  [[code](https://github.com/MinghanLi/MS-CSC-Rain-Streak-Removal)] [[google site](https://sites.google.com/view/cvpr-anonymity)][[youtube](https://www.youtube.com/watch?v=tYHX7q0yK4M)]

* Robust Video Content Alignment and Compensation for Rain Removal in a CNN Framework (2018 CVPR)
  * Chen Jie et al. Robust Video Content Alignment and Compensation for Rain Removal in a CNN Framework [[paper](https://arxiv.org/abs/1803.10433)]  [[Dr. Jie Chen's GIT (dataset available)](https://github.com/hotndy/SPAC-SupplementaryMaterials)] [[Prof. Lap-Pui Chau's homepage](http://www.ntu.edu.sg/home/elpchau/)]
  * Chen Jie et al. Robust Video Content Alignment and Compensation for Clear Vision Through the Rain [[paper](https://arxiv.org/abs/1804.09555)](*tips: I guess this is the extended journal version*)

* Erase or Fill? Deep Joint Recurrent Rain Removal and Reconstruction in Videos (2018 CVPR)
    * Liu Jiaying et al. Erase or Fill? Deep Joint Recurrent Rain Removal and Reconstruction in Videos [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Erase_or_Fill_CVPR_2018_paper.pdf)][[code](https://github.com/flyywh/J4RNet-Deep-Video-Deraining-CVPR-2018)][[Prof. Jiaying Liu's homepage](http://www.icst.pku.edu.cn/struct/people/liujiaying.html)] [[Dr. Wenhan Yang's homepage](http://www.icst.pku.edu.cn/struct/people/whyang.html)]


2017
--
* MoG: hould We Encode Rain Streaks in Video as Deterministic or Stochastic (2017 ICCV)
  * Wei Wei et al. Should We Encode Rain Streaks in Video as Deterministic or Stochastic? [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Wei_Should_We_Encode_ICCV_2017_paper.html)] 
[[code](https://github.com/wwxjtu/RainRemoval_ICCV2017)]

* FastDeRain: A novel tensor-based video rain streaks removal approach via utilizing discriminatively intrinsic priors (2017 CVPR)
  * Jiang Taixiang et al. A novel tensor-based video rain streaks removal approach via utilizing discriminatively intrinsic priors. [[paper](http://openaccess.thecvf.com/content_cvpr_2017/html/Jiang_A_Novel_Tensor-Based_CVPR_2017_paper.html)]
  * Fastderain: A novel video rain streak removal method using directional gradient priors. [[paper](https://arxiv.org/abs/1803.07487)][[code](https://github.com/TaiXiangJiang/FastDeRain)]


* Matrix decomposition: Video Desnowing and Deraining Based on Matrix Decomposition (2017 CVPR)
  * Ren Weilong et al. Video Desnowing and Deraining Based on Matrix Decomposition. [[paper](http://openaccess.thecvf.com/content_cvpr_2017/html/Ren_Video_Desnowing_and_CVPR_2017_paper.html)]

2015-2016
--
* Adherent raindrop modeling, detectionand removal in video (2016 TPAMI)
  * You Shaodi et al. Adherent raindrop modeling, detectionand removal in video. [[paper](https://ieeexplore.ieee.org/abstract/document/7299675/)] [[project page](http://www.cvl.iis.u-tokyo.ac.jp/~yousd/CVPR2013/Shaodi_CVPR2013.html "Not Available")]

* Video deraining and desnowing using temporal correlation and low-rank matrix completion (2015 TIP)
  * Kim JH et al. Video deraining and desnowing using temporal correlation and low-rank matrix completion. [[paper](https://ieeexplore.ieee.org/abstract/document/7101234/)] [[code](http://mcl.korea.ac.kr/~jhkim/deraining/)]  

* Utilizing local phase information to remove rain from video (2015 IJCV)
  * Santhaseelan et al. Utilizing local phase information to remove rain from video. [[paper](https://link.springer.com/article/10.1007/s11263-014-0759-8)] 
