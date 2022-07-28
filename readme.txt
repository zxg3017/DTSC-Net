mixmatch 本质上是通过数据增强的方式来增加新的伪标签数据, 并且对于生成的伪标签和真实标签会通过给予不同权重加以控制.
Dual-task Consistency 则是通过带标签的数据去学习主要信息, 在通过不同任务之间的一致性关系去保证不带标签的数据和带标签的数据能保持一致性, 从而提升效果.
目前将mixmatch作为数据增强的方式加入Dual-task Consistency中会有以下几个问题:
(1) mixmatch 之后, 真实的数据对(数据和标签)会被一定程度上和不带标签的数据融合, 导致真实数据的准确性有所下降, 这点可能会对Dual-task Consistency的任务有一定的影响
(2) mixmatch 对于未带标签的数据得到的伪标签并没有在Dual-task Consistency中起到作用
(3) loss层面完全沿用Dual-task Consistency的loss部分, 则无法控制生成的伪标签和真实标签之间的区分

针对以上几点可以尝试考虑的修改:
(1) 将数据划分成几个不同的部分, 1. 真实带标签的数据 2. 真实不带标签的数据 3. 真实带标签的数据经过mixmatch混合后的新数据 4. 真实不带标签的数据经过mixmatch混合后的新数据. 
(2) 1. 真实带标签的数据 用于计算 Dual-task Consistency 原始论文监督部分的loss
(3) 3. 真实带标签的数据经过mixmatch混合后的新数据 和 4. 真实不带标签的数据经过mixmatch混合后的新数据 则根据伪标签 用于计算 Dual-task Consistency 原始论文带监督部分的loss, 之后将这部分loss按照 mixmatch给定的权重作为(2)部分监督loss的额外部分
(4) 1, 2, 3, 4全部数据用于计算 Dual-task Consistency 原始论文不带监督部分的loss


