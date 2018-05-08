import tensorflow as tf

from network import Network

class ResAttentionNet(Network):
    """
    Implementation of Residual Attention Network model
    """
    def setup(self):
        # Pre_conv
        self.pre_conv = self.conv(self.inputs, 3, 3, 16, 1, 1, "pre_conv", relu=False)
        # Attention_1
        self.attention_1 = self.attention_module(self.pre_conv, 16, "attention_1")
        # Trans_1
        self.trans_1 = self.residual_unit(self.attention_1, 16, 16, "trans_2", stride=1)
        # Attention_2
        self.attention_2 = self.attention_module(self.trans_1, 16, "attention_2")
        # Trans_2
        self.trans_2 = self.residual_unit(self.attention_2, 16, 32, "trans_3", stride=2)
        # Attention_3
        self.attention_3 = self.attention_module(self.trans_2, 32, "attention_3")
        # Post_res_1
        self.post_res_1 = self.residual_unit(self.attention_3, 32, 64, "post_res_1", stride=2)
        # Post_res_2
        self.post_res_2 = self.residual_unit(self.post_res_1, 64, 64, "post_res_2")
        # Post_res_3
        self.post_res_3 = self.residual_unit(self.post_res_2, 64, 64, "post_res_3")
        # Post_res_4
        self.post_res_4 = self.residual_unit(self.post_res_3, 64, 64, "post_res_4")
        # Post_bn
        self.post_bn = self.batch_normal(self.post_res_4, self.is_train, "post_bn", tf.nn.relu) 
        # Ave
        self.post_pool = self.avg_pool(self.post_bn, 8, 8, 1, 1, "post_pool")
        # raw_score
        self.raw_score = self.fc(self.post_pool, 10, "raw_score", relu=False)
        # score
        self.score = self.softmax(self.raw_score, "score")

    def attention_module(self, x, ci, name, p=1, t=2, r=1):
        """
        Implementation of Attention Module (2 pool in soft mask branch)
        Input:
        --- x: Module input, 4-D Tensor, with shape [bsize, height, width, channel]
        --- ci: Input channels
        --- name: Module name
        --- p: The number of pre-processing Residual Units
        --- t: The number of Residual Units in trunk branch
        --- r: The number of Residual Units between adjacent pooling layer in the mask branch
        Output:
        --- outputs: Module output
        """
        with tf.name_scope(name), tf.variable_scope(name):
            # Pre-processing Residual Units
            with tf.name_scope("pre_processing"), tf.variable_scope("pre_processing"):
                pre_pros = x
                for idx in range(p):
                    unit_name = "pre_res_{}".format(idx + 1)
                    pre_pros = self.residual_unit(pre_pros, ci, ci, unit_name)
            # Trunk branch
            with tf.name_scope("trunk_branch"), tf.variable_scope("trunk_branch"):
                trunks = pre_pros
                for idx in range(t):
                    unit_name = "trunk_res_{}".format(idx + 1)
                    trunks = self.residual_unit(trunks, ci, ci, unit_name)
            # Mask branch
            with tf.name_scope("mask_branch"), tf.variable_scope("mask_branch"):
                size_1 = pre_pros.get_shape().as_list()[1:3]
                # Max pooling
                masks_1 = self.max_pool(pre_pros, 3, 3, 2, 2, "pool_1")
                for idx in range(r):
                    unit_name = "mask_res1_{}".format(idx + 1)
                    masks_1 = self.residual_unit(masks_1, ci, ci, unit_name)
                size_2 = masks_1.get_shape().as_list()[1:3]
                # Max pooling
                masks_2 = self.max_pool(masks_1, 3, 3, 2, 2, "pool_2")
                for idx in range(2 * r):
                    unit_name = "mask_res2_{}".format(idx + 1)
                    masks_2 = self.residual_unit(masks_2, ci, ci, unit_name)
                # Interpolation
                masks_3 = self.upsample(masks_2, "inter_1", size_2)
                # Skip connection
                skip = self.residual_unit(masks_1, ci, ci, "skip")
                masks_3 = tf.add(masks_3, skip, name="fuse_add")
                for idx in range(r):
                    unit_name = "mask_res3_{}".format(idx + 1)
                    masks_3 = self.residual_unit(masks_3, ci, ci, unit_name)
                # Interpolation
                masks_4 = self.upsample(masks_3, "inter_2", size_1)
                # Batch Normalization
                masks_4 = self.batch_normal(masks_4, self.is_train, "mask_bn1", tf.nn.relu)
                # 1x1 Convolution
                masks_4 = self.conv(masks_4, 1, 1, ci, 1, 1, "mask_conv1", relu=False)
                # Batch Normalization
                masks_4 = self.batch_normal(masks_4, self.is_train, "mask_bn2", tf.nn.relu)
                # 1x1 Convolution
                masks_4 = self.conv(masks_4, 1, 1, ci, 1, 1, "mask_conv2", relu=False)
                # Sigmoid
                masks_4 = tf.nn.sigmoid(masks_4, "mask_sigmoid")
            # Fusing
            with tf.name_scope("fusing"), tf.variable_scope("fusing"):
                outputs = tf.multiply(trunks, masks_4, name="fuse_mul")
                outputs = tf.add(trunks, outputs, name="fuse_add")
            # Post-processing Residual Units
            with tf.name_scope("post_processing"), tf.variable_scope("post_processing"):
                for idx in range(p):
                    unit_name = "post_res_{}".format(idx + 1)
                    outputs = self.residual_unit(outputs, ci, ci, unit_name)
        return outputs

    def residual_unit(self, x, ci, co, name, stride=1):
        """
        Implementation of Residual Unit
        Input:
        --- x: Unit input, 4-D Tensor, with shape [bsize, height, width, channel]
        --- ci: Input channels
        --- co: Output channels
        --- name: Unit name
        --- stride: Convolution stride
        Output:
        --- outputs: Unit output
        """
        with tf.name_scope(name), tf.variable_scope(name):
            # Batch Normalization
            bn_1 = self.batch_normal(x, self.is_train, "bn_1", tf.nn.relu)
            # 1x1 Convolution
            conv_1 = self.conv(bn_1, 1, 1, co/4, 1, 1, "conv_1", relu=False)
            # Batch Normalization
            bn_2 = self.batch_normal(conv_1, self.is_train, "bn_2", tf.nn.relu)
            # 3x3 Convolution
            conv_2 = self.conv(bn_2, 3, 3, co/4, stride, stride, "conv_2", relu=False)
            # Batch Normalization
            bn_3 = self.batch_normal(conv_2, self.is_train, "bn_3", tf.nn.relu)
            # 1x1 Convolution
            conv_3 = self.conv(bn_3, 1, 1, co, 1, 1, "conv_3", relu=False)
            # Skip connection
            if co != ci or stride > 1:
                skip = self.conv(bn_1, 1, 1, co, stride, stride, "conv_skip", relu=False)
            else:
                skip = x
            outputs = tf.add(conv_3, skip, name="fuse")
            return outputs
