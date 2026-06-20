# 关键技术点3：端侧 VLM 驱动的自然语言具身任务规划与安全执行

![Edge VLM](https://img.shields.io/badge/Edge%20VLM-Cosmos%20Reason2%202B-6A1B9A)
![ROS2](https://img.shields.io/badge/ROS2-Humble-22314E)
![YOLOE](https://img.shields.io/badge/YOLOE--26-Candidate%20Generation-FF6F00)
![Nav2](https://img.shields.io/badge/Nav2-Safe%20Execution-1565C0)

本部分展示端侧 VLM 如何从单纯图文问答模块进入真实移动机器人任务闭环。系统不让 VLM 直接控制底盘，而是将其限定在自然语言理解、开放语义目标裁判和结果生成环节；机器人动作由本地技能映射、安全校验、任务状态机和 Nav2 导航链路执行。

## 核心思想

| 设计点 | 作用 |
|---|---|
| VLM 语义理解 | 将自然语言任务解析为结构化 intent / steps |
| YOLO 高频候选 | 持续生成可定位、可接近的目标候选框 |
| VLM 低频裁判 | 对颜色、姿态、手持物和空间关系等开放语义条件进行判断 |
| 本地安全执行 | 将模型输出约束到白名单技能、合法航点、可取消任务和 Nav2 执行链路 |
| 任务反馈闭环 | 通过语音播报、场景解说和报告生成记录任务过程 |

## 总体链路

```mermaid
flowchart TD
    A["用户语音 / 文本输入"] --> B["语音识别与文本入口<br/>voice_input_node.py"]
    B --> C["自然语言任务解析<br/>nl_parser_node.py"]

    C --> D1["VLM Planner<br/>vlm_client.py<br/>vlm_planner.py<br/>vlm_intent_parser.py<br/>nl_prompt.py"]
    C --> D2["多步任务切分<br/>task_segmenter.py"]
    C --> D3["结构化语义 Schema<br/>nlu_schema.py"]

    D1 --> E["本地安全校验与技能映射<br/>intent_validator.py<br/>skill_mapper.py"]
    D2 --> E
    D3 --> E

    E --> F["统一执行出口<br/>task_executor.py"]
    F --> G["任务状态机调度<br/>Mission Director"]

    G --> H["Nav2 导航到观测点"]
    G --> I["YOLO 高频候选生成"]
    I --> J["候选框编号与过滤<br/>person_semantic_matcher_node.py"]

    J --> K["VLM 三阶段语义裁判<br/>semantic_match_prompt.py"]
    K --> K1["阶段1：独立视觉事实观察"]
    K --> K2["阶段2：用户条件拆解"]
    K --> K3["阶段3：候选编号匹配判断"]

    K3 --> L{"是否匹配目标"}
    L -- "否" --> M["继续搜索 / 换观测角度 / 任务中止"]
    L -- "是" --> N["目标接近 / 执行动作"]

    N --> O["场景解说与语音反馈<br/>scene_narrator_node.py<br/>tts_node.py"]
    O --> P["任务记录与报告生成<br/>mission_logger_node.py"]

    classDef input fill:#E3F2FD,stroke:#1565C0,color:#0D47A1,stroke-width:1.5px;
    classDef vlm fill:#F3E5F5,stroke:#6A1B9A,color:#4A148C,stroke-width:1.5px;
    classDef safe fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20,stroke-width:1.5px;
    classDef out fill:#FFF3E0,stroke:#EF6C00,color:#E65100,stroke-width:1.5px;
    class A,B input;
    class C,D1,D2,D3,J,K,K1,K2,K3 vlm;
    class E,F,G,H,I,L,M,N safe;
    class O,P out;
```

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/01_算法链路说明.md`](./docs/01_算法链路说明.md) | 说明自然语言任务解析、VLM 三阶段语义裁判和本地安全执行链路 |
| [`docs/02_关键源码索引.md`](./docs/02_关键源码索引.md) | 说明源码文件职责与核心链路 |

## 源码范围

本目录整理端侧语义理解、候选目标裁判、技能映射、安全执行、语音反馈和任务报告相关的核心实现，便于沿着源码查看自然语言任务进入机器人执行闭环的过程。
