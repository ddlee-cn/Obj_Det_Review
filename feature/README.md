
# （四）目标检测领域的新趋势之特征复用、实时性

本篇关注检测模型的头部部分。在图片经过基础网络映射后，获得的特征如何有效地利用在检测中，是一个中心问题。另外，本篇也介绍一部分面向实时检测的工作。

## 特征复用与整合

### FPN

[Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)

对图片信息的理解常常关系到对位置和规模上不变性的建模。在较为成功的图片分类模型中，Max-Pooling这一操作建模了位置上的不变性：从局部中挑选最大的响应，这一响应在局部的位置信息就被忽略掉了。而在规模不变性的方向上，添加不同大小感受野的卷积核（VGG），用小卷积核堆叠感受较大的范围（GoogLeNet），自动选择感受野的大小（Inception）等结构也展现了其合理的一面。

回到检测任务，与分类任务不同的是，检测所面临的物体规模问题是跨类别的、处于同一语义场景中的。

一个直观的思路是用不同大小的图片去生成相应大小的feature map，但这样带来巨大的参数，使本来就只能跑个位数图片的内存更加不够用。另一个思路是直接使用不同深度的卷积层生成的feature map，但较浅层的feature map上包含的低等级特征又会干扰分类的精度。

本文提出的方法是在高等级feature map上将特征向下回传，反向构建特征金字塔。

![fpn](img/fpn.png)

从图片开始，照常进行级联式的特征提取，再添加一条回传路径：从最高级的feature map开始，向下进行最近邻上采样得到与低等级的feature map相同大小的回传feature map，再进行元素位置上的叠加（lateral connection），构成这一深度上的特征。

这种操作的信念是，低等级的feature map包含更多的位置信息，高等级的feature map则包含更好的分类信息，将这两者结合，力图达到检测任务的位置分类双要求。

特征金字塔本是很自然的想法，但如何构建金字塔同时平衡检测任务的定位和分类双目标，又能保证显存的有效利用，是本文做的比较好的地方。如今，FPN也几乎成为特征提取网络的标配，更说明了这种组合方式的有效性。

### DSSD

[Deconvolutional Single Shot Multibox Detector]()

本文是利用反卷积操作对SSD的改进。

![dssd](img/dssd.png)

在原版SSD中，检测头部不仅从基础网络提取特征，还添加了额外的卷积层，而本文则在这些额外卷积层后再添加可学习的反卷积层，并将feature map的尺度扩展为原有尺寸，并将两个方向上具有相同尺度的feature map叠加后再进行检测，这种设计使检测头部同时利用不同尺度上的低级特征和高级特征。




### RON

[RON: Reverse Connection with Objectness Prior Networksfor Object Detection](https://arxiv.org/abs/1707.01691)

![ron](img/ron.png)

文章关注两个问题：1)多尺度目标检测，2）正负样本比例失衡的问题。

对于前者，文章将相邻的feature map通过reverse connection相连，并在每个feature map上都进行检测，最后再整合过滤。对于后者，类似RPN，对每个anchor box生成一个Objectness priori，作为一个指标来过滤过多的box（但不对box进行调整，RPN对box进行调整，作者指出这会造成重复计算）。文章的实验中RON在较低的分辨率下取得了超过SSD的表现。

### RefineDet

[Single-Shot Refinement Neural Network for Object Detection](https://arxiv.org/abs/1711.06897)

![RefineDet](img/refinedet.jpg)

本文是一阶段的模型，但思路上却是两阶段的。文章指出两阶段方法精度有优势的原因有三点：1）两阶段的设计使之有空间来用采样策略处理类别不均衡的问题；2）级联的方式进行box回归；3）两阶段的特征描述。文章提出两个模块来在一阶段检测器中引入两阶段设计的优势：Anchor Refinement Module(ARM)和Object Detection Module(ODM)。前者用于识别并过滤背景类anchor来降低分类器的负担，并且调整anchor位置以更好的向分类器输入，后者用于多分类和box的进一步回归。

Single-shot的体现在上面两个模块通过Transfer Connection Block共用特征。除此之外，Transfer Connection Block还将特征图反传，构成类似FPN的效果。两个模块建立联合的loss使网络能够端到端训练。


实验结果显示RefineNet的效果还是不错的，速度跟YOLOv2相当，精度上更有优势。之后的Ablation experiments也分别支撑了负样本过滤、级联box回归和Transfer Connection Block的作用。可以说这篇文章的工作让两阶段和一阶段检测器的界限更加模糊了。


## 面向实时性的工作

### Light Head R-CNN

[Light-Head R-CNN: In Defense of Two-Stage Object Detector](https://arxiv.org/abs/1711.07264)

文章指出两阶段检测器通常在生成Proposal后进行分类的“头”(head)部分进行密集的计算，如ResNet为基础网络的Faster-RCNN将整个stage5（或两个FC）放在RCNN部分， R-FCN要生成一个具有随类别数线性增长的channel数的Score map，这些密集计算正是两阶段方法在精度上领先而在推断速度上难以满足实时要求的原因。

针对这两种元结构(Faster-RCNN和RFCN)，文章提出了“头”轻量化方法，试图在保持精度的同时又能减少冗余的计算量，从而实现精度和速度的Trade-off。

![light-head](img/light-head.png)

如上图，虚线框出的部分是三种结构的RCNN子网络（在每个RoI上进行的计算），light-head R-CNN中，在生成Score map前，ResNet的stage5中卷积被替换为sperable convolution，产生的Score map也减少至10×p×p（相比原先的#class×p×p）。

一个可能的解释是，“瘦”（channel数较少）的score map使用于分类的特征信息更加紧凑，原先较“厚”的score map在经过PSROIPooling的操作时，大部分信息并没有提取（只提取了特定类和特定位置的信息，与这一信息处在同一score map上的其他数据都被忽略了）。

进一步地，位置敏感的思路将位置性在channel上表达出来，同时隐含地使用了更类别数相同长度的向量表达了分类性（这一长度相同带来的好处即是RCNN子网络可以免去参数）。

light-head在这里的改进则是把这一个隐藏的嵌入空间压缩到较小的值，而在RCNN子网络中加入FC层再使这个空间扩展到类别数的规模，相当于是把计算量分担到了RCNN子网络中。

粗看来，light-head将原来RFCN的score map的职责两步化了：thin score map主攻位置信息，RCNN子网络中的FC主攻分类信息。另外，global average pool的操作被去掉，用于保持精度。

### SSDLite(MobileNets V2)
