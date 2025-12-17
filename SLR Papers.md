# SLR Papers

## Table of Content
- **[Papers](#papers)**
  - [Skeleton SLR](#skeletonslr)
  - [RGB SLR](RGB)

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
