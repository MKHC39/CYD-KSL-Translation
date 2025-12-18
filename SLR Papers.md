# SLR Papers

## Table of Content
- **[Papers](#papers)**
  - [Skeleton SLR](#skeleton-slr)
  - [RGB SLR](#rgb-slr)
  - [KSL Papers](#ksl-papers)
  - [Others](#others)
  
- **[Datasets](#datasets)**

- **[Summary of Results](#Summary)**

## Papers

### Skeleton SLR
1. **Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison** [*paper*](https://arxiv.org/pdf/1910.11006)
1. **Pose-based Sign Language Recognition using GCN and BERT** [*paper*](https://arxiv.org/pdf/2012.00781)
1. **SignBERT: Pre-Training of Hand-Model-Aware Representation for Sign Language Recognition** [*paper*](https://arxiv.org/pdf/2110.05382)
1. **OpenHands: Making Sign Language Recognition Accessible with Pose-based Pretrained Models across Languages** [*paper*](https://aclanthology.org/2022.acl-long.150.pdf)
1. **Skeleton Aware Multi-modal Sign Language Recognition** [*paper*](https://arxiv.org/pdf/2103.08833)
1. **SIGN LANGUAGE RECOGNITION BASED ON HAND AND BODY SKELETAL DATA** [*paper*](https://vcl.iti.gr/media/documents/sign-language-recognition-based-on-hand-and-body-skeletal-data.pdf)
1. **Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition** [*paper*](https://ojs.aaai.org/index.php/AAAI/article/view/12328)

### RGB SLR
1. **A Comprehensive Study on Deep Learning-based Methods for Sign Language Recognition** [*paper*](https://arxiv.org/pdf/2007.12530)
1. **Dynamic Sign Language Recognition Based on Video Sequence With BLSTM-3D Residual Networks** [*paper*](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8667292)
1. **Recurrent Convolutional Neural Networks for Continuous Sign Language Recognition by Staged Optimization** [*paper*](https://openaccess.thecvf.com/content_cvpr_2017/papers/Cui_Recurrent_Convolutional_Neural_CVPR_2017_paper.pdf)
1. **Video-Based Sign Language Recognition Without Temporal Segmentation** [*paper*](https://ojs.aaai.org/index.php/AAAI/article/view/11903)
1. **Dilated Convolutional Network with Iterative Optimization for Continuous Sign Language Recognition** [*paper*](https://www.ijcai.org/proceedings/2018/0123.pdf)
1. **Continuous Sign Language Recognition with Correlation Network** [*paper*](https://openaccess.thecvf.com/content/CVPR2023/papers/Hu_Continuous_Sign_Language_Recognition_With_Correlation_Network_CVPR_2023_paper.pdf)
1. **Continuous Sign Language Recognition With Multi-Scale Spatial-Temporal Feature Enhancement** [*paper*](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10829616)
1. **Fully Convolutional Networks for Continuous Sign Language Recognition** [*paper*](https://arxiv.org/pdf/2007.12402)


### KSL Papers
1. **딥러닝 기반 OpenPose를 이용한 한국 수화 동작 인식에 관한 연구** [*paper*](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE10549905&buildDate=2025-11-11+16%3A57%3A59&nowDate=20251209_2&cdnUrl=https%3A%2F%2Fcdn.dbpia.co.kr%2Fstatic&buildTime=20251111165759&minify=.min&appVersion=1.0.0&language=ko_KR&hasTopBanner=true)
1. **실시간 수어 AI 번역 프로그램 구현** [*paper*](http://journal.dcs.or.kr/_common/do.php?a=full&b=12&bidx=3464&aidx=38387)
1. **미디어파이프와 장단기기억을 이용한 수화 동작인식 앱 개발** [*paper*](http://journal.dcs.or.kr/_common/do.php?a=current&b=11&bidx=3205&aidx=35772)
1. **Deep Neural Network-Based Sign Language Translation Model with Dynamic Coordinate Compression** [*paper*](https://koreascience.kr/article/JAKO202523836005515.pdf)


### Others
1. A Comprehensive Study on Deep Learning-based Methods for Sign Language Recognition [*paper*](https://arxiv.org/pdf/2007.12530)
1. A Comprehensive Review of Sign Language Recognition: Different Types, Modalities, and Datasets [*paper*](https://arxiv.org/pdf/2204.03328)

## KSL Datasets?
1. https://www.kci.go.kr/kciportal/landing/article.kci?arti_id=ART002867515
1. https://arxiv.org/pdf/1811.11436

## Datasets

| Dataset                                                      | Language    | Classes | Samples | Data Type                    | Language Level |
| ------------------------------------------------------------ | ----------- | ------- | ------- | ---------------------------- | :------------- |
| **[NationalCSL-DP](https://figshare.com/articles/media/NationalCSL-DP/27261843)** | Chinese      | 6,707   | 134,140   | Videos                        | isolated    |
| **[CE-CSL](https://pan.baidu.com/s/1OHJLRfLFPWqkxvLBr4KAQg?from=init#list/path=%2F)** | Chinese      | 5,988   | 5,988   | Videos                        | continuous    |
| **[NMFs-CSL](https://ustc-slr.github.io/datasets/2020_nmfs_csl/)** | Chinese      | 1,067   | 1,067   | Videos                        | isolated    |
| **[DEVISIGN](https://vipl.ict.ac.cn/homepage/ksl/data.html)** | Chinese      | 4,414   | 331,050   | Videos & Depth from Kinect                       | isolated    |
| **[RWTH-PHOENIX-Weather 2014](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/)** | German      | 1,081   | 6,841   | Videos                       | continuous     |
| [**RWTH-PHOENIX-Weather 2014 T**](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/) | German      | 1,066   | 8,257   | Videos                       | continuous     |
| **[ASLLVD](http://www.bu.edu/asllrp/av/dai-asllvd.html)**    | American    | 3,300   | 9,800   | Videos(multiple angles)      | isolated       |
| **[ASLLVD-Skeleton](https://www.cin.ufpe.br/~cca5/asllvd-skeleton/)** | American    | 3,300   | 9,800   | Skeleton                     | isolated       |
| **[SIGNUM](https://www.phonetik.uni-muenchen.de/forschung/Bas/SIGNUM/)** | German      | 450     | 33,210  | Videos                       | continuous     |
| [**DEVISIGN-G**](http://vipl.ict.ac.cn/homepage/ksl/data.html) | Chinese     | 36      | 432     | Videos                       | isolated       |
| [**DEVISIGN-D**](http://vipl.ict.ac.cn/homepage/ksl/data.html) | Chinese     | 500     | 6,000   | Videos                       | isolated       |
| [**DEVISIGN-L**](http://vipl.ict.ac.cn/homepage/ksl/data.html) | Chinese     | 2000    | 24,000  | Videos                       | isolated       |
| [**GSL isol.**](https://vcl.iti.gr/dataset/gsl/)             | Greek       | 310     | 40,785  | Videos&Depth from RealSense  | isolated       |
| [**GSL SD**](https://vcl.iti.gr/dataset/gsl/)                | Greek       | 310     | 10,290  | Videos&Depth from RealSense  | continuous     |
| [**GSL SI**](https://vcl.iti.gr/dataset/gsl/)                | Greek       | 310     | 10,290  | Videos&Depth from RealSense  | continuous     |
| [**IIITA -ROBITA**](https://robita.iiita.ac.in/dataset.php)  | Indian      | 23      | 605     | Videos                       | isolated       |
| [**PSL Kinect**](http://vision.kia.prz.edu.pl/dynamickinect.php) | Polish      | 30      | 300     | Videos&Depth from Kinect     | isolated       |
| [**PSL ToF**](http://vision.kia.prz.edu.pl/dynamictof.php)   | Polish      | 84      | 1,680   | Videos&Depth from ToF camera | isolated       |
| [**LSE-Sign**](http://lse-sign.bcbl.eu/web-busqueda/)        | Spanish     | 2,400   | 2,400   | Videos                       | isolated       |
| [**MS-ASL**](https://www.microsoft.com/en-us/research/publication/ms-asl-a-large-scale-data-set-and-benchmark-for-understanding-american-sign-language/)        | American    | 1000   | 25,513   | Videos                       | isolated       |
| [**Purdue RVL-SLLL**](https://engineering.purdue.edu/RVL/Database/ASL/asl-database-front.htm) | American    | 39      | 546     | Videos                       | isolated       |
| [**RWTH-BOSTON-50**](http://www-i6.informatik.rwth-aachen.de/aslr/database-rwth-boston-50.php) | American    | 50      | 483     | Videos(multiple angles)      | isolated       |
| [**RWTH-BOSTON-104**](http://www-i6.informatik.rwth-aachen.de/aslr/database-rwth-boston-104.php) | American    | 104     | 201     | Videos(multiple angles)      | continuous     |
| [**WLASL**](https://dxli94.github.io/WLASL/)                 | American    | 2,000   | 21,083  | Videos                       | isolated       |
| [**NIASL2021**](https://aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&searchKeyword=%EC%88%98%EC%96%B4%EB%B2%88%EC%97%AD&aihubDataSe=data&dataSetSn=103)                 | Korean    | 6,000   | 536,000  | Videos                       | mixed      |
| [**NIASLG1?**](https://aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&searchKeyword=%EC%88%98%EC%96%B4%EB%B2%88%EC%97%AD&aihubDataSe=data&dataSetSn=636)                 | Korean    | 164,375   | 201,026  | Videos                       | continuous       |
| [**AUTSL**](https://cvml.ankara.edu.tr/datasets/)            | Turkish     | 226     | 38,336    | Videos & Depth from Kinect  | isolated |

## Summary
- **[Table of results](https://docs.google.com/spreadsheets/d/1GMigSrAvbKFR-B0k2Yud1iNjaEvC-MK_sDVotKH0x-M/edit?usp=sharing)**
