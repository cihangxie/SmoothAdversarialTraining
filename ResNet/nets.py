# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from adv_model import AdvImageNetModel
from resnet_model import (
    resnet_group, resnet_bottleneck, resnet_backbone)


NUM_BLOCKS = {
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3]
}


class ResNetModel(AdvImageNetModel):
    def __init__(self, args):
        self.num_blocks = NUM_BLOCKS[args.depth]
        self.group = args.group
        self.res2_bottleneck = args.res2_bottleneck
        self.activation_name = args.activation_name.lower()

    def get_logits(self, image):

        def block_func(l, ch_out, stride):
            l = resnet_bottleneck(l, ch_out, stride, group=self.group, res2_bottleneck=self.res2_bottleneck, activation_name=self.activation_name)
            return l

        return resnet_backbone(image, self.num_blocks, resnet_group, block_func, activation_name=self.activation_name)
