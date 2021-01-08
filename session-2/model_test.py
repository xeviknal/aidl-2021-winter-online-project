import unittest
import torch
from model import ConvBlock
from model import MyLeNet


class TestMyModel(unittest.TestCase):
    def test_convolutional_block(self):
        conv = ConvBlock(1, 6, 5, 2)
        x = torch.randn(1, 1, 64, 64)
        self.assertEqual(conv(x).shape, torch.Size([1, 6, 30, 30]))

    def test_my_lenet_model(self):
        model = MyLeNet()
        x = torch.randn(1, 1, 64, 64)
        self.assertEqual(model(x).shape, torch.Size([1, 15]))


if __name__ == '__main__':
    unittest.main()
