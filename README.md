#### 一. 任务形式化

这个赛道的目标是设立基于知识图谱的主动聊天任务。[知识驱动对话-官网地址](http://lic2019.ccf.org.cn/talk)

##### 输入: 

对话目标g，其中g=START->TOPIC\_A->TOPIC\_B;表示机器从冷启动状态主动聊到话题A，然后聊到话题B。意味着在该任务中，由机器主动的引导对话；在该任务中，具体的话题包括电影和娱乐人物主体。

相关知识信息M，其中M=f1,f2,...,fn. 包括三类，分别是：话题A的知识信息，话题B的知识信息，话题A和话题B的关联信息。在该任务中，具体的相关知识信息包括电影票房，导演和评价等，以SPO形式表示。也就是(Subject, Predicate, Object)，即(实体一，谓词，实体二)。

当前对话序列H=u1,u2,...u(t-1)

##### 输出：

机器回复ut.

#### 二.数据介绍(见官方网站)

#### 三.评价方法

自动评估指标和人工评估指标结合。自动评估指标考虑三个层面的度量，分别是字级别(F1-score)，词级别(BLEU)和回复多样性(DISTINCT)。关于回复多样性，还是基于词的计算，不过考察的是生成词的另外一个维度。在参考2中作者这样写道：

_distinct-1 and distinct-2
are respectively the number of distinct unigrams and bigrams divided by total number of generated words_

#### 四.一般流程

参考PyTorch官方提供的tutorial(见参考4)，从seq2seq的角度解决问题的方法是，将多轮对话拆分成平行句。例如，针对当前对话序列H=u1,u2,...u(t-1)，可以拆分成t-2组样本，分别是：u1->u2;u2->u3;...;u(t-2)->u(t-1);但是，这样的划分方式存在明显的问题是：句子之间的平滑。这应该是一个问题，但是还没有深入思考过。

#### 五.想法实现

目前，重构了PyTorch官方的Chatbot的Tutorial代码，将各个模块解耦出来，顺带发现了一个Bug。在此基础上，准备实现一个baseline，[代码地址](https://github.com/zhpmatrix/lic2019-competition)这里，但是目前还不清楚怎样将知识图谱的信息添加到pipeline中。

参考：

1.[第六届全国社会媒体处理大会-SMP2017中文人机对话技术评测(ECDT)](http://ir.hit.edu.cn/SMP2017-ECDT)

包含两个任务：用户意图领域分类和特定域任务型人机对话在线评测

2.《A Diversity-Promoting Objective Function for Neural Conversation Models》

3.《A Persona-Based Neural Conversation Model》

4.[chatbot tutorial with pytorch](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html)
