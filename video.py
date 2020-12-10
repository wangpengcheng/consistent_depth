#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import cv2
import logging
import os
from os.path import join as pjoin
import sys
import tempfile
from utils import (frame_sampling, image_io)
from utils.helpers import mkdir_ifnotexists


ffmpeg = "ffmpeg"
ffprobe = "ffprobe"

# 查找相似的图片帧
def sample_pairs(frame_range, flow_ops):
    #TODO: update the frame range with reconstruction range
    name_mode_map = frame_sampling.SamplePairsMode.name_mode_map()
    opts = [
        frame_sampling.SamplePairsOptions(mode=name_mode_map[op]) for op in flow_ops
    ]
    pairs = frame_sampling.SamplePairs.sample(
        opts, frame_range=frame_range, two_way=True
    )
    print(f"Sampled {len(pairs)} frame pairs.")
    return pairs

# 基础视频抽象类
class Video:
    def __init__(self, path, video_file=None):
        self.path = path
        self.video_file = video_file

    def check_extracted_pts(self):
        pts_file = "%s/frames.txt" % self.path
        if not os.path.exists(pts_file):
            return False
        with open(pts_file, "r") as file:
            lines = file.readlines()
            self.frame_count = int(lines[0])
            width = int(lines[1])
            height = int(lines[2])
            print("%d frames detected (%d x %d)." % (self.frame_count, width, height))
            if len(lines) != self.frame_count + 3:
                sys.exit("frames.txt has wrong number of lines")
            print("frames.txt exists, checked OK.")
            return True
        return False

    def extract_pts(self):
        if self.check_extracted_pts():
            # frames.txt exists and checked OK.
            return
        # 检查是否存在文件
        if not os.path.exists(self.video_file):
            sys.exit("ERROR: input video file '%s' not found.", self.video_file)

        # Get width and height
        tmp_file = tempfile.mktemp(".png")
        cmd = "%s -i %s -vframes 1 %s" % (ffmpeg, self.video_file, tmp_file)
        print(cmd)
        res = os.popen(cmd).read()
        image = image_io.load_image(tmp_file)
        height = image.shape[0]
        width = image.shape[1]
        os.remove(tmp_file) # 删除中间文件
        if os.path.exists(tmp_file):
            sys.exit("ERROR: unable to remove '%s'" % tmp_file)

        # Get PTS
        def parse_line(line, token):
            if line[: len(token)] != token:
                sys.exit("ERROR: record is malformed, expected to find '%s'." % token)
            return line[len(token) :]
        # 查询视频帧以及关键时间点
        ffprobe_cmd = "%s %s -select_streams v:0 -show_frames" % (
            ffprobe,
            self.video_file,
        )
        cmd = ffprobe_cmd + " | grep pkt_pts_time"
        print(cmd)
        res = os.popen(cmd).read()
        pts = []
        for line in res.splitlines():
            pts.append(parse_line(line, "pkt_pts_time="))
        self.frame_count = len(pts)
        print("%d frames detected." % self.frame_count)
        # 设置帧输出文件
        pts_file = "%s/frames.txt" % self.path
        with open(pts_file, "w") as file:
            file.write("%d\n" % len(pts))
            file.write("%s\n" % width)
            file.write("%s\n" % height)
            for t in pts:
                file.write("%s\n" % t)
        # 再次确认文件
        self.check_extracted_pts()

    def check_frames(self, frame_dir, extension, frames=None):
        if not os.path.isdir(frame_dir):
            return False
        # 获取文件夹中的文件
        files = os.listdir(frame_dir)
        files = [n for n in files if n.endswith(extension)]
        if len(files) == 0:
            return False
        # 随机选帧进行生成
        if frames is None:
            frames = range(self.frame_count)

        if len(files) != len(frames):
            sys.exit(
                "ERROR: expected to find %d files but found %d in '%s'"
                % (self.frame_count, len(files), frame_dir)
            )
        for i in frames:
            frame_file = "%s/frame_%06d.%s" % (frame_dir, i, extension)
            if not os.path.exists(frame_file):
                sys.exit("ERROR: did not find expected file '%s'" % frame_file)
        print("Frames found, checked OK.")

        return True
    # 将视频扩展成为一张张的图片
    def extract_frames(self):
        frame_dir = "%s/color_full" % self.path # 设置输出文件夹
        mkdir_ifnotexists(frame_dir)
        # 检查frame是否已经存在
        if self.check_frames(frame_dir, "png"):
            # Frames are already extracted and checked OK.
            return

        if not os.path.exists(self.video_file):
            sys.exit("ERROR: input video file '%s' not found.", self.video_file)
        # 不存在的话，就生成关键帧
        cmd = "%s -i %s -start_number 0 -vsync 0 %s/frame_%%06d.png" % (
            ffmpeg,
            self.video_file,
            frame_dir,
        )
        print(cmd)
        os.popen(cmd).read()

        count = len(os.listdir(frame_dir))
        if count != self.frame_count:
            sys.exit(
                "ERROR: %d frames extracted, but %d PTS entries."
                % (count, self.frame_count)
            )
        # 再次进行帧确认
        self.check_frames(frame_dir, "png")
    # 对所有帧进行缩放
    def downscale_frames(
        self, subdir, max_size, ext, align=16, full_subdir="color_full"
    ):
        full_dir = pjoin(self.path, full_subdir)
        down_dir = pjoin(self.path, subdir)
        # 创建缩放的文件夹
        mkdir_ifnotexists(down_dir)
        # 检查缩放文件夹是否存在
        if self.check_frames(down_dir, ext):
            # Frames are already extracted and checked OK.
            return
        # 检查所有帧
        for i in range(self.frame_count):
            full_file = "%s/frame_%06d.png" % (full_dir, i)
            down_file = ("%s/frame_%06d." + ext) % (down_dir, i)
            suppress_messages = (i > 0)
            # 加载图片
            image = image_io.load_image(
                full_file, max_size=max_size, align=align,
                suppress_messages=suppress_messages
            )
            image = image[..., ::-1]  # Channel swizzle
            # 根据加载格式进行数据修改
            if ext == "raw":
                image_io.save_raw_float32_image(down_file, image)
            else:
                cv2.imwrite(down_file, image * 255)

        self.check_frames(down_dir, ext)
