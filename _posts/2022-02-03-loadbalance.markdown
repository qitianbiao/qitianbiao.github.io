---
layout: post
title:  "负载均衡 Loadbalance"
date:   2022-02-03 2:00:00 +0000
categories: jekyll update
---
# 负载均衡

早先提供服务的软件是没有负载均衡概念的，如果软件所在的服务器经过观察测量发现负载过大而出现明显的性能劣化，往往是通过增加服务器的硬件配置来提高处理更多负载的能力，这就是通常所说的`纵向扩展`。随着分布式系统模式逐渐成为软件系统市场的主流体系，提供一种服务的软件系统实例的个数增加的越来越多，人们发现这种`横向扩展`的模式提供的性能提高方案更廉价，管理维护手段更灵活，为了协调这些实例使它们在其对外表现上像是一个单一的服务提供者，负载均衡器自然而然的出现了，它矗立于业务软件的多实例前面成为统一对外提供服务的入口和门户。
![lb]({{ site.url }}/images/loadbalance_1.png)

## 负载均衡的定义

负载均衡load balance是指将访问特定服务器或服务的流量负载分散到提供该服务的一组服务器或服务的方法，负载均衡可以降低单一服务的压力，并提升整体服务的吞吐量。实现负载均衡的应用软件或整套软硬件系统被称为负载均衡器load balancer。

## 负载均衡带来的好处

### 1. 使用分布式系统架构必然会使用负载均衡
客户端访问一个分布式系统提供的服务，必然要解决访问哪个服务端实例，怎么访问该实例的问题，负载均衡就是这个问题的答案。基于谁来决定访问实例，可以分为客户端负载均衡、服务端负载均衡；基于在数据的那一层来进行负载均衡的，可以分为七层/应用层负载均衡、四层/非应用层负载均衡。

### 2. 防止服务过载失效，提升整体服务吞吐量、可靠性和可用性

把过量的负载分散到各个服务器或服务上，单一服务器或服务的负载就可以降低了，而且单一服务器或服务失效不会导致整体对外服务的失效，从而保证了服务整体的可靠性和高可用性。

### 3. 易管理，易运维
使用负载均衡后，服务器或服务实例可以很方便的进行水平扩展，只需要添加新服务器或服务的访问方式和地址即可，扩展的过程中用户也感知不到服务的中断。企业本身也可以测量负载量和服务器或者服务数量的比例，预测未来的资源采购计划，达到资源管理和成本控制的目的。

## 负载均衡的分类
### 硬件、软件
硬件负载均衡器是一个单独的网络设备，和硬件交换机、硬件路由器、硬件防火墙进行组网互联互通后，专门完成业务数据流量的负载分发任务。因为使用了专用的集成电路和处理芯片，性能优异。
软件负载均衡器不可触摸不可见，是一个做负载均衡用途的应用程序。常见的软件负载均衡器需要在操作系统之上进行安装部署，或者以容器的形式运行，也有的软件负载均衡器是以操作系统内核组件的形式直接提供给用户配置使用。

### 四层、七层
根据OSI网络七层模型，四层为数据传输层，常见的协议有TCP/UDP；七层为应用层，常见的协议有HTTP。处理四层或四层以下数据报文的负载均衡器称之为四层负载均衡器，越往下层的负载均衡性能一般是越好；处理七层数据报文的负载均衡器称之为七层负载均衡器，越往上层的负载均衡功能会越来越丰富。

### 转发、代理
转发最早出现于描述网络数据报文，路由器接收到数据报文后根据报文头里描述的目的IP地址匹配路由表后进行从一个接口到另一个借口的转发，转发类型的负载均衡器也是类似的概念，只不过关注的对象是服务本身而不是报文，事实上四层负载均衡都是转发模式，对二层的帧或三层的报文进行修整后往后面转发，之所以叫四层负载均衡，主要是从服务本身“IP地址加端口号”的通用表现形式而定义的。
代理不同于转发的区别在于，代理工作在应用层面，所以它首先要终结客户端过来的TCP连接，并代替客户端往服务器端发起新的TCP连接，在这个过程中实现包括了业务流量转发之外的更多功能，可以理解为七层的负载均衡都是代理模式。从客户端的角度出发，代理又可以分为如下几种：
- 正向代理，代理客户端，客户端需要自己配置代理，服务器端看到的业务流量来自于代理。
- 透明代理，拦截并处理客户端的业务流量，客户端无需自己配置代理，服务器端看到的业务流量来自于代理，但服务器端可以看到客户端的真实IP地址。
- 反向代理，代理服务器端，客户端看到的业务流量来自于代理。

### 服务端/集中式、客户端/分散式
传统的负载均衡器都是服务端的，从通信距离上看更贴近于服务端，有些负载均衡器软件通常是和服务软件部署在同一子网甚至于同一服务器上，适用的场景是大量的客户端访问少量的服务端这种情景。
随着微服务模式的兴起，拆分后的微服务之间产生了更多更复杂的访问流量，同一微服务既是服务端提供特定服务，也有可能是客户端访问其它的微服务，因此传统的集中式负载均衡只能解决服务集群外部进来流量（南北流量）的均衡，而内部服务之间的流量（东西流量）的均衡，需要客户端负载均衡器来实现。

## 负载均衡器典型的解决方案

 | 名称 | 分类 | 介绍 | 
 | ------ | ------ | ------ | 
 | F5 | 硬件，独立系统形式，四层/七层，转发/代理，服务端 | 
 | LVS | 软件，操作系统内核形式，四层，转发，服务端 |  | 
 | HAProxy+KeepAlived | 软件，应用程序形式，四层/七层，转发/代理，服务端 | 
 | Nginx | 软件，应用程序形式，七层，代理，服务端 | 
 | Istio+Envoy | 软件，容器边车形式，四层/七层，转发/代理，客户端 | 

## 负载均衡的工作原理

### 服务端负载均衡，四层转发
四层转发模式的负载均衡关注的是数据报文本身，要解决的问题是怎么把业务数据报文以负载均衡的策略转发到后端服务器主机上，然后操作系统会自然而然的根据服务器上特定服务监听的端口找到对应的软件服务本身。四层转发模式的负载均衡，在TCP层面上是透明的，即客户端和服务器端之间只有一条TCP通道。而根据二层/三层处理数据报文的方法不同，四层负载均衡器有以下几种模式：

#### 直接路由/三角传输/单臂模式
![lb]({{ site.url }}/images/loadbalance_2.png)
负载均衡器LB和所有的服务器SV都要设置为同一个虚拟IP地址VIP，客户端CL发送对SV的VIP的请求后，经由LB来决定在某一个时刻该VIP对应的二层MAC地址是哪个SV的，并修改请求报文的二层MAC，由此来实现报文转发。特定SV收到业务数据报文后，会使用VIP直接响应客户端CL，而不经过LB。

#### 隧道模式
![lb]({{ site.url }}/images/loadbalance_3.png)
负载均衡器LB和所有的服务器SV都要设置为同一个虚拟IP地址VIP，客户端CL发送对SV的VIP的请求后，经由LB来决定在某一个时刻转发给哪个SV的真实IP，并对请求报文添加一层包头，在包头里描述目的地址为SV的真实IP，由此来实现报文转发。特定SV收到业务数据报文后，会首先去掉那层包头，并使用VIP直接响应客户端CL，而不经过LB。

#### NAT模式
![lb]({{ site.url }}/images/loadbalance_4.png)
负载均衡器LB设置为网关，客户端CL发送对LB的请求后，经由LB来决定在某一个时刻转发给哪个SV的真实IP，并修改请求报文的目的IP为该SV的真实IP（DNAT模式），由此来实现报文转发。特定SV收到业务数据报文后，会将响应交还给网关LB，再经由LB修改报文的源IP为LB的IP返回给CL。

### 服务端负载均衡，七层代理
七层代理模式的负载均衡关注的是应用层数据，最常用的也就是HTTP报文，除了能像四层负载均衡一样可以策略的均衡转发请求外，七层负载均衡可以实现更智能化和更强大的功能。七层代理模式的负载均衡，在TCP层面上维护和客户端以及和服务端的两条不同的TCP通道。

#### 反向代理
反向代理近服务端，代理的是服务端多实例的流量，客户端看到的是唯一的服务入口，正因为如此，可以实现服务端负载均衡的功能，例如典型的Nginx。反向代理还可以在此基础上，实现更多的功能，比如根据客户发出请求的URL地址的不一样，路由到不同的服务端（网关）；或根据session的不一样，进行亲和性的服务调用，从而实现HTTP的长连接；根据具体的业务特征，实现服务细颗粒度的治理逻辑等。
![lb]({{ site.url }}/images/loadbalance_5.png)

#### 和负载均衡无关的其他代理模式：正向代理，透明代理
正向代理位于客户端附近，客户端需要配置代理的信息因此客户端是可以感知的，它主要代理客户端的HTTP请求，实现诸如访问原来没办法访问到的资源（客户端无权限，但代理有），对客户端访问服务行为的认证和授权（代理会询问用户名和密码），对服务端隐藏客户端信息的安全举措（服务端只看到是代理IP发来的请求，但是注意代理本身知道所有一切信息）。
透明代理类似于正向代理的位置和作用，但客户端不需要配置，因此客户端无感知。另外，和正向代理不同的是，服务端也对代理也无感知，服务端看到的HTTP请求仍然来自于客户端，实际上是发自代理并且又可能已经被修改，因此透明代理类似于成功的“中间人攻击”角色，透明代理通常配置在内网的路由器或者ISP供应商，属于上帝模式。

### 客户端负载均衡
客户端负载均衡器是在传统服务拆分成若干个微服务的背景下产生的。拆分后的微服务既是其它服务的客户端也是自己的服务端，服务之间需要相互调用，而且单个服务依然有多实例的存在，因此如何将服务调用的流量均衡的转发，将是客户端负载均衡需要解决的问题。客户端的负载均衡器位于“扮演客户端”的服务实例“内部”，和服务实例本身共享资源。最早的客户端负载均衡的实现源于Java系SpringClould框架，负载均衡器的实现是在应用的服务代码里通过直接引用实现的，运行时中它们其实是同一进程。
![lb]({{ site.url }}/images/loadbalance_6.png)

作为“服务端”的服务需要将自己所有实例的访问路径“注册”进注册中心，由注册中心来维护这些实例的状态更新和上下线。而作为“客户端”的服务在需要访问”服务端“服务时，先查询注册中心，获取信息后，由自己的负载均衡器决定以何种负载均衡的方式访问”服务端“服务。

#### 边车代理
边车代理是客户端负载均衡的一种进化模式。“边车”的理念来自于Kubernets的一个Pod内可以有包含一个主要提供服务的容器之外的多个容器的设计，这些容器共享同一网络和文件命名空间。在边车代理模式的负载均衡中，服务应用和负载均衡器是独立的容器，已经解耦，但共享同一个Pod的资源，它们之间的交互是通过网络loopback设备进行，也非常高效。
![lb]({{ site.url }}/images/loadbalance_7.png)

所有服务实例都会搭配一个边车，边车拦截所有出入服务的应用流量，并根据控制平面例如Istio的配置管理，对途径流量进行治理。边车会通过xDS发现协议动态获取它需要的所有服务实例连接信息和状态等信息，以及控制平面下发的配置内容例如需要使用的负载均衡策略。

## 负载均衡的策略/算法
如何实现负载的绝对活相对均衡是一个复杂的问题，因为实际环境相当复杂，而且用户需求和服务的资源需求也各不相同。到目前为止，常用的负载均衡策略有：

 | 策略 | 描述 | 说明 | 
 | ------ | ------ | ------ | 
 | 随机 | 第一个请求随机分配给某个服务器/服务实例，第二个请求也随机分给某个服务器/服务实例，随机性效果取决于随机算法本身。 | 
 | 轮询 | 第一个请求分配给第一个服务器/服务实例，第二个请求分给第二个服务器/服务实例，以此到最后再从第一个服务器/服务实例循环。 | 
 | 加权 | 一般用于真实的服务器，根据服务器的硬件配置不同，其处理请求的性能也不一样，高性能的服务器配置的权重更高，被分配到请求的可能性也更大。 | 
 | 最小连接数 | 监控当前所有服务器/服务实例正在处理的请求连接个数，把下一个请求分配给那个最小连接个数也是最小负荷的服务服务器/服务实例。 | 
 | 一致性Hash | 确保请求负载会均匀分布于所有服务器/服务实例中，确保相同特征的请求每次都能分配给相同的服务器/服务实例，一致性强调服务器/服务实例在扩所容或节点故障时，并不影响其它服务器/服务实例的hash值。服务和请求按照特定特征计算hash值后共同形成一个hash环，请求会被发送给离它顺时针最近的服务。<br>请求的hash计算可能是MAC地址，IP地址，请求的服务名加参数，用户的ID等中的某一种，服务器/服务实例的hash计算一般都是IP地址和端口号。 | 

## 负载均衡的级联、并联
负载均衡器是可以级联多级的，每级负载均衡实现自己效能的最大化，例如集中式的负载均衡级联，一般四层负载均衡器放在七层负载均衡器前面，或者硬件负载均衡器在软件负载均衡之前。

负载均衡器特别是软件负载均衡器可以部署多个并联起来，实现对外服务资源的区域划分和特色管控。
而部署多个同类型的负载均衡并联当作一个负载均衡使用时，可以实现负载均衡自己的高可用性，消除单点故障。例如：HAProxy负责负载均衡，提高服务器均的性能和扩展性，而Keepalived工作机制类似于直接路由模式的四层负载均衡，来负责HAProxy的高可用性。

## 负载均衡的附加价值
因为负载均衡器天然具备识别报文内容的能力，所以在不同的网络层面，它可以叠加很多的管理功能，这使得负载均衡器再也不仅仅是只做负载均衡的机器。

### 负载均衡+网关/负载均衡+路由中心
负载均衡关注于对请求流量如何均衡的转发给后端服务，而网关/路由中心关注于根据请求内容的特征转发给后端服务。从通信位置上，集中式的负载均衡器和网关/路由中心实际上是重叠的，现实中的产品也是一体的，例如基于Nginx这个反向代理模式负载均衡器而开发的Kubernetes官方服务网关Ingress Controller。
如果一定要分开网关和负载均衡的话，请求流量应该先经过网关进行路由，然后再经过负载均衡器实现负载均衡转发。负载均衡器和网关都具备识别请求流量特征的能力，因此它们都可以附加过滤、认证授权、流量治理、监控、缓存等高级功能。

### 负载均衡+缓存服务器
负载均衡器可以附带缓存特定的服务常用或热点数据例如静态资源等，实现快速回复，提供更快的请求响应，降低后端服务的压力。

### 负载均衡+防火墙
负载均衡可以附加安全策略实现对典型服务攻击例如DDoS，SQL注入等安全防护。

### 负载均衡+服务治理解决方案
可以在七层的负载均衡器层面直接进行服务熔断，降级，镜像流量，故障注入等治理。

