import unittest

from PIL.JpegImagePlugin import JpegImageFile
from torchvision import transforms
from dataset import MyDataset


class TestingMyDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dataset = MyDataset(images_path="./data/test/data/",
                                 labels_path="./data/test/chinese_mnist_test.csv",
                                 transform=self.transform)

    def test_init(self):
        self.assertEqual(self.dataset.transform, self.transform)
        self.assertEqual(self.dataset.images_path, "./data/test/data/")
        self.assertTrue(len(self.dataset.labels_csv) > 1)

    def test_len(self):
        self.assertEqual(len(self.dataset), len(self.dataset.labels_csv))

    def test_get_item(self):
        image, code = self.dataset[0]
        self.assertEqual(code, 9)
        self.assertIsInstance(image, JpegImageFile, 'Sir, that is inappropriate.')


if __name__ == '__main__':
    unittest.main()
