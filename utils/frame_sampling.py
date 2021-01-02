#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from collections import namedtuple
from enum import Enum, unique, auto
from typing import Iterable, NamedTuple, Dict, Any, Set
import numpy as np

from .frame_range import FrameRange


@unique
class SamplePairsMode(Enum):
    EXHAUSTED = 0
    CONSECUTIVE = auto()
    HIERARCHICAL = auto()
    HIERARCHICAL2 = auto()

    @classmethod
    def name_mode_map(cls):
        return {v.name.lower(): v for v in cls}

    @classmethod
    def names(cls):
        return [v.name.lower() for v in cls]


# param is default to {} while mode is required
class SamplePairsOptions(NamedTuple):
    mode: SamplePairsMode  # 模型数据
    params: Dict[str, Any] = {}  # 存储的名称和字典


# 定义基础数据集合
Pair = namedtuple("Pair", ["first", "second"])
Pairs_t = Set[Pair]  # 设置属性集合


# 相似对构造函数
class SamplePairs:
    @classmethod
    def sample(
        cls,
        opts: Iterable[SamplePairsOptions],
        frame_range: FrameRange,
        two_way=False,
    ) -> Pairs_t:
        num_frames = len(frame_range)

        rel_pairs = set()
        # 创建真实的相似对
        for opt in opts:
            rel_pairs = rel_pairs.union(cls.factory(num_frames, opt, two_way))

        pairs = set()
        for rel_pair in rel_pairs:
            pair = Pair(frame_range.index_to_frame[rel_pair[0]],
                        frame_range.index_to_frame[rel_pair[1]])
            # Filter out pairs where no end is in depth_frames. Can be optimized
            # when constructing these pairs
            if (pair[0] in frame_range.frames()
                    or pair[1] in frame_range.frames()):
                pairs.add(pair)
        return pairs

    # 静态工厂方法，用来创建对应的函数选择对象
    @classmethod
    def factory(cls, num_frames: int, opt: SamplePairsOptions,
                two_way: bool) -> Pairs_t:
        # 函数对应hash表
        funcs = {
            SamplePairsMode.EXHAUSTED: cls.sample_exhausted,
            SamplePairsMode.CONSECUTIVE: cls.sample_consecutive,
            SamplePairsMode.HIERARCHICAL: cls.sample_hierarchical,
            SamplePairsMode.HIERARCHICAL2: cls.sample_hierarchical2,
        }
        # 执行对应的函数
        return funcs[opt.mode](num_frames, two_way, **opt.params)

    # 静态方法，查找相似等级函数
    @staticmethod
    def sample_hierarchical(
        num_frames: int,
        two_way: bool,
        min_dist=1,
        max_dist=None,
        include_mid_point=False,
    ) -> Pairs_t:
        """
        Args:
            min_dist, max_dist: minimum and maximum distance to the neighbour
        """
        assert min_dist >= 1
        # 计算最大最远距离
        if max_dist is None:
            max_dist = num_frames - 1
        # 设置最小等级，主要是对最小距离求2的对数
        min_level = np.ceil(np.log2(min_dist)).astype(int)
        # 设置最大等级
        max_level = np.floor(np.log2(max_dist)).astype(int)
        # 设置每个阶梯的等级，如果含有中间值时，查找最大值
        step_level = (lambda l: max(0, l - 1)) if include_mid_point else (
            lambda l: l)
        # 设置signs
        signs = (-1, 1) if two_way else (1, )
        # set集合
        pairs = set()
        # 遍历所有可能间隔
        for level in range(min_level, max_level + 1):
            # 计算对应的距离
            dist = 1 << level
            # 计算步长
            step = 1 << step_level(level)
            # 按照步长计算所有帧
            for start in range(0, num_frames, step):
                # 遍历前后范围(-1,1)
                for sign in signs:
                    # 计算尾部
                    end = start + sign * dist
                    if end < 0 or end >= num_frames:
                        continue
                    # 将对帧数据添加上去
                    pairs.add(Pair(start, end))
        return pairs

    # 修饰静态方法，查找相似等级2--注意这里含有中间值点
    @classmethod
    def sample_hierarchical2(cls,
                             num_frames: int,
                             two_way: bool,
                             min_dist=1,
                             max_dist=None) -> Pairs_t:
        return cls.sample_hierarchical(
            num_frames,
            two_way,
            min_dist=min_dist,
            max_dist=max_dist,
            include_mid_point=True,
        )

    @classmethod
    def sample_consecutive(cls, num_frames: int, two_way: bool) -> Pairs_t:
        return cls.sample_hierarchical(num_frames,
                                       two_way,
                                       min_dist=1,
                                       max_dist=1)

    @staticmethod
    def sample_exhausted(cls, num_frames: int, two_way: bool) -> Pairs_t:
        second_frame_range = ((lambda i, N: range(N)) if two_way else
                              (lambda i, N: range(i + 1, N)))

        pairs = set()
        for i in range(num_frames):
            for j in second_frame_range(i, num_frames):
                if i != j:
                    pairs.add(Pair(i, j))
        return pairs

    @classmethod
    def to_one_way(cls, pairs) -> Pairs_t:
        def ordered(pair):
            if pair[0] > pair[1]:
                return Pair(*pair[::-1])
            return Pair(*pair)

        return {ordered(p) for p in pairs}


def to_in_range(pairs, frame_range=None):
    if frame_range is None:
        return pairs

    def in_range(idx):
        return frame_range[0] <= idx and idx < frame_range[1]

    return [pair for pair in pairs if all(in_range(i) for i in pair)]
