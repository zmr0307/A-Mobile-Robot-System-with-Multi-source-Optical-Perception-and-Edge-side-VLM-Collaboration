# 基于多源光感知与端侧 VLM 协同的移动机器人系统

![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-22314E)
![Jetson Orin NX](https://img.shields.io/badge/Jetson-Orin%20NX-76B900)
![FAST-LIO2](https://img.shields.io/badge/FAST--LIO2-Localization-2E7D32)
![Nav2](https://img.shields.io/badge/Nav2-Safe%20Navigation-1565C0)
![TensorRT](https://img.shields.io/badge/TensorRT-YOLOE--26-FF6F00)
![Edge VLM](https://img.shields.io/badge/Edge%20VLM-Cosmos%20Reason2%202B-6A1B9A)

本仓库用于展示第二十一届中国研究生电子设计竞赛作品中的关键技术点源码与算法改进链路。当前整理内容包括**关键技术点1：免初值全局重定位**、**关键技术点2：多源几何-语义安全导航**和**关键技术点3：端侧 VLM 驱动的自然语言具身任务规划与安全执行**。

## 关键技术点

| 序号 | 技术点 | 核心链路 | 入口 |
|---|---|---|---|
| 1 | 免初值全局重定位 | BBS 全局搜索 + NDT 三维复核 + 多帧稳定投票 + FAST-LIO2 固定偏移接管 | [`关键技术点1_免初值全局重定位/`](./关键技术点1_免初值全局重定位/) |
| 2 | 多源几何-语义安全导航 | L2 / D435i 几何融合 + YOLOE-26 语义代价地图 + Clearance A* + MPPI 横向约束 | [`关键技术点2_多源几何语义安全导航/`](./关键技术点2_多源几何语义安全导航/) |
| 3 | 端侧 VLM 驱动的自然语言具身任务规划与安全执行 | 自然语言解析 + YOLO 候选生成 + VLM 语义裁判 + 本地安全执行 + 任务报告闭环 | [`关键技术点3_端侧VLM驱动的自然语言具身任务规划与安全执行/`](./关键技术点3_端侧VLM驱动的自然语言具身任务规划与安全执行/) |

## 系统链路概览

本系统不是三个孤立算法模块的简单拼接，而是围绕真实移动机器人任务形成一条端侧闭环：先解决机器人任意位置启动后的全局位姿接入，再将多源光感知结果转化为安全导航代价，最后把自然语言任务和开放语义目标判断接入可验证、可中断的机器人执行链路。

```mermaid
flowchart LR
    subgraph HW["硬件与传感器层"]
        H1["Unitree L2<br/>三维激光雷达"]
        H2["RealSense D435i<br/>RGB-D 相机"]
        H3["Wheeltec N100<br/>IMU"]
        H4["麦克风 / 扬声器<br/>语音交互"]
        H5["WL100<br/>四驱四转向底盘"]
    end

    subgraph LOC["关键技术点1：免初值全局重定位"]
        L1["当前点云输入"] --> L2["BBS 全局盲搜<br/>多候选生成"]
        L2 --> L3["NDT 三维复核<br/>异常候选剔除"]
        L3 --> L4["多帧稳定投票<br/>Bridge 二次确认"]
        L4 --> L5["FAST-LIO2 接管<br/>固定 map→odom 偏移"]
    end

    subgraph NAV["关键技术点2：多源几何-语义安全导航"]
        N1["L2 点云<br/>大范围几何结构"] --> N4["多层代价地图<br/>几何 + 语义融合"]
        N2["D435i 深度点云<br/>近场低矮障碍"] --> N4
        N3["YOLOE-26<br/>语义目标检测"] --> N4
        N4 --> N5["Clearance A*<br/>中线偏好全局规划"]
        N5 --> N6["MPPI Omni<br/>横向采样约束"]
    end

    subgraph VLM["关键技术点3：端侧 VLM 具身任务闭环"]
        V1["语音 / 文本<br/>自然语言输入"] --> V2["VLM Planner<br/>任务意图解析"]
        V2 --> V3["结构化任务队列<br/>intent / steps / action"]
        V4["YOLO 高频候选<br/>目标框编号"] --> V5["VLM 三阶段裁判<br/>视觉事实 / 条件拆解 / 候选匹配"]
        V3 --> V6["本地安全执行<br/>技能映射 + 状态机"]
        V5 --> V6
    end

    subgraph OUT["执行与反馈层"]
        O1["Nav2 导航执行"] --> O2["目标接近 / 搜索 / 归位"]
        O2 --> O3["TTS 语音反馈"]
        O3 --> O4["任务日志与报告生成"]
    end

    H1 --> LOC
    H1 --> NAV
    H2 --> NAV
    H2 --> VLM
    H3 --> LOC
    H4 --> VLM
    H5 --> OUT

    L5 --> N4
    N6 --> O1
    V6 --> O1

    classDef hardware fill:#ECEFF1,stroke:#546E7A,color:#263238,stroke-width:1.5px;
    classDef loc fill:#E3F2FD,stroke:#1565C0,color:#0D47A1,stroke-width:1.5px;
    classDef nav fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20,stroke-width:1.5px;
    classDef vlm fill:#F3E5F5,stroke:#6A1B9A,color:#4A148C,stroke-width:1.5px;
    classDef out fill:#FFF3E0,stroke:#EF6C00,color:#E65100,stroke-width:1.5px;
    classDef group fill:#FAFAFA,stroke:#BDBDBD,color:#424242,stroke-width:1px;

    class H1,H2,H3,H4,H5 hardware;
    class L1,L2,L3,L4,L5 loc;
    class N1,N2,N3,N4,N5,N6 nav;
    class V1,V2,V3,V4,V5,V6 vlm;
    class O1,O2,O3,O4 out;
    class HW,LOC,NAV,VLM,OUT group;
```

## 三个技术点之间的关系

| 层级 | 解决的问题 | 输出给下一层的能力 |
|---|---|---|
| 定位层 | 机器人在任意位置启动后缺少全局位姿 | 稳定的 `map -> odom` 对齐关系和可用于导航的全局位姿 |
| 感知导航层 | 单一传感器难以同时覆盖低矮障碍、语义目标和通道安全裕度 | 融合几何与语义风险的代价地图、安全居中的全局路径和稳定局部控制 |
| 具身任务层 | 自然语言和开放语义目标难以直接落地为机器人动作 | 结构化任务、候选目标裁判结果、可执行动作和任务反馈记录 |

## 评委阅读路径

| 阅读目标 | 建议入口 | 重点查看 |
|---|---|---|
| 看定位算法改进 | [`关键技术点1_免初值全局重定位/`](./关键技术点1_免初值全局重定位/) | BBS 多候选、NDT 复核、多帧稳定投票、FAST-LIO2 固定偏移接管 |
| 看感知导航安全性 | [`关键技术点2_多源几何语义安全导航/`](./关键技术点2_多源几何语义安全导航/) | L2 / D435i / YOLOE 融合、Clearance A*、MPPI 横向约束 |
| 看 VLM 如何落地到机器人任务 | [`关键技术点3_端侧VLM驱动的自然语言具身任务规划与安全执行/`](./关键技术点3_端侧VLM驱动的自然语言具身任务规划与安全执行/) | VLM Planner、YOLO 候选编号、三阶段语义裁判、本地安全执行 |
