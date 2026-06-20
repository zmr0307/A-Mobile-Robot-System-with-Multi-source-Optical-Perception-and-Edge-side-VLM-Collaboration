# 关键技术点2：多源几何-语义安全导航

![Nav2](https://img.shields.io/badge/Nav2-Costmap%20Fusion-1565C0)
![YOLOE-26](https://img.shields.io/badge/YOLOE--26-TensorRT-FF6F00)
![Clearance A*](https://img.shields.io/badge/Clearance%20A*-Global%20Planner-2E7D32)
![MPPI](https://img.shields.io/badge/MPPI-Omni%20Lateral%20Constraint-6A1B9A)

## 技术目标

本技术点面向移动机器人在窄通道、低矮障碍物、动态目标和复杂局部环境中的安全导航问题。系统将 Unitree L2 三维激光雷达和 Intel RealSense D435i RGB-D 相机形成的几何感知，与 YOLOE-26 TensorRT 语义检测结果融合到 Nav2 代价地图中，并结合 Clearance A* 全局规划和 MPPI Omni 横向约束，实现兼顾障碍物安全距离、语义目标规避和全向底盘稳定控制的导航链路。

> **边界说明**
> 本技术点关注“感知信息如何进入导航决策”和“规划控制如何约束安全运动”。其中 VLM 任务理解、自然语言解析和任务编排属于后续关键技术点，不混入本目录；本目录只保留几何感知、语义代价地图、安全规划、MPPI 横向约束和 Nav2 集成相关源码。

## 链路概览

| 层级 | 输入 | 输出 | 核心源码 |
|---|---|---|---|
| 几何感知 | Unitree L2 点云、D435i 深度点云、静态地图 | 体素障碍物与膨胀代价 | `nav2_params.yaml` |
| 语义感知 | D435i RGB、D435i aligned depth | YOLO 检测框、类别、距离 | `yolo_trt_node.cpp` |
| 语义代价地图 | 检测结果、深度距离、TF | Nav2 costmap 语义障碍物 | `semantic_detection_layer.cpp` |
| 安全全局规划 | 全局 costmap | 更远离障碍的全局路径 | `clearance_a_star_planner.cpp` |
| 局部控制 | 全局路径、局部 costmap、机器人状态 | 平滑速度指令 | `nav2_params.yaml`、`dynamic_vy_std_guard.py` |

## 总体流程

```mermaid
flowchart TD
    subgraph A["多源几何输入"]
        A1["① Unitree L2 点云"]
        A2["② D435i Depth PointCloud2"]
        A3["③ 静态栅格地图"]
    end

    subgraph B["视觉语义感知"]
        B1["④ D435i RGB 图像"]
        B2["⑤ YOLOE-26 TensorRT 推理"]
        B3["⑥ 深度 ROI 距离估计"]
        B4["⑦ Detection2DArray"]
        B1 --> B2 --> B4
        A2 --> B3 --> B4
    end

    subgraph C["融合代价地图"]
        C1["⑧ SpatioTemporalVoxelLayer"]
        C2["⑨ SemanticDetectionLayer"]
        C3["⑩ InflationLayer"]
        C1 --> C3
        C2 --> C3
    end

    subgraph D["安全全局规划"]
        D1["⑪ Clearance Field"]
        D2["⑫ Clearance A*"]
        D3["⑬ 安全居中路径"]
        D1 --> D2 --> D3
    end

    subgraph E["MPPI Omni 局部控制"]
        E1["⑭ MPPI 轨迹采样"]
        E2["⑮ 横向速度 / vy_std 约束"]
        E3["⑯ Critics 组合评分"]
        E4["⑰ Velocity Smoother"]
        E1 --> E2 --> E3 --> E4
    end

    A1 --> C1
    A2 --> C1
    A3 --> C3
    B4 --> C2
    C3 --> D1
    D3 --> E1

    classDef geom fill:#E3F2FD,stroke:#1565C0,color:#0D47A1,stroke-width:1.5px;
    classDef semantic fill:#FFF3E0,stroke:#EF6C00,color:#E65100,stroke-width:1.5px;
    classDef costmap fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20,stroke-width:1.5px;
    classDef planner fill:#F3E5F5,stroke:#6A1B9A,color:#4A148C,stroke-width:1.5px;
    classDef control fill:#FCE4EC,stroke:#AD1457,color:#880E4F,stroke-width:1.5px;
    classDef group fill:#FAFAFA,stroke:#BDBDBD,color:#424242,stroke-width:1px;
    class A1,A2,A3 geom;
    class B1,B2,B3,B4 semantic;
    class C1,C2,C3 costmap;
    class D1,D2,D3 planner;
    class E1,E2,E3,E4 control;
    class A,B,C,D,E group;
```

## 核心改进

1. 使用 D435i RGB-D 与 Unitree L2 点云共同参与 Nav2 代价地图构建，弥补单一传感器对低矮、近距离或语义目标感知不足的问题。
2. 使用 YOLOE-26 TensorRT C++ 节点在 Jetson Orin NX 上进行实时语义检测，并结合深度 ROI 估计目标距离。
3. 设计 SemanticDetectionLayer，将语义检测目标投影到 Nav2 costmap，使视觉识别到的目标能够直接影响导航避障。
4. 使用 SpatioTemporalVoxelLayer 融合 L2 点云和 D435i 点云，并通过时间衰减机制降低动态障碍残影。
5. 设计 Clearance A* 全局规划器，根据障碍物距离场对贴墙、贴障路径施加代价惩罚，使路径更倾向于通道中部。
6. 在 MPPI Omni 局部控制中约束横向采样强度，通过 `vy_std` 和动态调节脚本减少全向底盘横向抖动，同时保留近目标微调能力。

## 目录说明

| 路径 | 内容 |
|---|---|
| [`docs/01_算法链路说明.md`](./docs/01_算法链路说明.md) | 多源几何-语义安全导航完整链路 |
| [`docs/02_关键源码索引.md`](./docs/02_关键源码索引.md) | 关键源码文件与算法环节对应关系 |
| [`src/`](./src/) | 本技术点对应的核心源码、插件配置和启动文件 |
