#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用语义目标匹配节点入口。

实现仍在 person_semantic_matcher_node.py 中，以保留旧接口兼容。
"""

from wl100_demo.person_semantic_matcher_node import main as _matcher_main


def main(args=None):
    _matcher_main(args=args, node_name="semantic_target_matcher")


if __name__ == "__main__":
    main()
