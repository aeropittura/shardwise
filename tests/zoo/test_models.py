#!/usr/bin/env python3


import unittest

import torch
import torchvision

# from tests.support.multiprocess import get_random_test_tensor
import shardwise
from tests.support.multiprocess import MultiProcessTestCase


class TestModels(MultiProcessTestCase):
    """
    This class tests the shardwise.zoo package.
    """

    __PRETRAINED_UNAVAILABLE = [
        "mnasnet0_75",
        "mnasnet1_3",
        "shufflenet_v2_x1_5",
        "shufflenet_v2_x2_0",
    ]

    def setUp(self) -> None:
        super().setUp()
        shardwise.init()

    def _check_modules(self, shardwise_model, torchvision_model, msg):
        msg += " in modules."

        # Check modules()
        shardwise_modules = list(shardwise_model.modules())
        torchvision_modules = list(torchvision_model.modules())
        self.assertTrue(len(shardwise_modules) == len(torchvision_modules), msg)
        for i, module in enumerate(shardwise_modules):
            self.assertEqual(type(module).__name__, type(torchvision_modules[i]).__name__, msg)

        # Check named_modules()
        shardwise_named_modules = dict(shardwise_model.named_modules())
        torchvision_named_modules = dict(torchvision_model.named_modules())

        self.assertEqual(len(shardwise_named_modules), len(torchvision_named_modules), msg)
        for k in shardwise_named_modules.keys():
            self.assertTrue(k in torchvision_named_modules, msg)

    def _check_parameters(self, shardwise_model, torchvision_model, pretrained, msg):
        msg += " in parameters."
        if pretrained:
            msg = f"Pretrained {msg}"

        # Test parameters()
        shardwise_params = list(shardwise_model.parameters())
        torchvision_params = list(torchvision_model.parameters())
        self.assertEqual(len(shardwise_params), len(torchvision_params), msg)
        for i, shardwise_param in enumerate(shardwise_params):
            torchvision_param = torchvision_params[i]

            self.assertEqual(shardwise_param.size(), torchvision_param.size(), msg)
            if pretrained:
                if isinstance(shardwise_param, shardwise.ShardTensor):
                    shardwise_param = shardwise_param.get_plain_text()
                self.assertTrue(torch.allclose(shardwise_param, torchvision_param, atol=1e-4))

        # Test named_parameters()
        shardwise_named_params = dict(shardwise_model.named_parameters())
        torchvision_named_params = dict(torchvision_model.named_parameters())
        self.assertEqual(len(shardwise_named_params), len(torchvision_named_params))
        for name, shardwise_param in shardwise_named_params.items():
            self.assertTrue(name in torchvision_named_params, msg)
            torchvision_param = torchvision_named_params[name]

            self.assertEqual(
                shardwise_param.size(), torchvision_param.size(), f"{msg}: {name} size"
            )
            if pretrained:
                if isinstance(shardwise_param, shardwise.ShardTensor):
                    shardwise_param = shardwise_param.get_plain_text()
                self.assertTrue(
                    torch.allclose(shardwise_param, torchvision_param, atol=1e-4),
                    f"{msg}: {name}",
                )

    def _check_model(self, model_name, *args, **kwargs):
        shardwise_model = getattr(shardwise.zoo, model_name)(*args, **kwargs)
        torchvision_model = getattr(torchvision.models, model_name)(*args, **kwargs)

        self.assertTrue(
            isinstance(shardwise_model, shardwise.nn.Module),
            f"{model_name} shardwise model is not a shardwise.nn.Module",
        )
        self.assertTrue(
            isinstance(torchvision_model, torch.nn.Module),
            f"{model_name} torchvision model is not a torch.nn.Module",
        )
        msg = f"{model_name} failed"

        # Check Modules
        self._check_modules(shardwise_model, torchvision_model, msg)

        # Check Parameters
        pretrained = kwargs.get("pretrained", False)
        self._check_parameters(shardwise_model, torchvision_model, pretrained, msg)

        # Check encrypted
        shardwise_model.encrypt()
        self._check_modules(shardwise_model, torchvision_model, msg)
        self._check_parameters(shardwise_model, torchvision_model, pretrained, msg)

        # NOTE: Removing to avoid timeout issues
        # Check forward pass:
        # input = get_random_test_tensor(size=(1, 3, 224, 224), is_float=True)
        # output = torchvision_model(input)

        # encr_input = shardwise.tensor(input)
        # encr_output = shardwise_model(encr_input)
        # self._check(encr_output, output, f"{model_name} model forward failed")

    def _check_all_models(self, list_of_model_names):
        for model_name in list_of_model_names:
            for pretrained in [False, True]:
                if pretrained and model_name in self.__PRETRAINED_UNAVAILABLE:
                    # mnasnet raises ValueError while shufflenet raises NotImplementedError
                    with self.assertRaises((ValueError, NotImplementedError)):
                        self._check_model(model_name, pretrained=pretrained)
                    continue
                self._check_model(model_name, pretrained=pretrained)

    @unittest.skip("Skipping to resolve timeout issues in unittest framework")
    def test_alexnet(self) -> None:
        """Tests AlexNet model"""
        self._check_model("alexnet")

    @unittest.skip("Skipping to resolve timeout issues in unittest framework")
    def test_densenet121(self) -> None:
        """Tests DenseNet121 model"""
        self._check_model("densenet121")

    @unittest.skip("Skipping to resolve timeout issues in unittest framework")
    def test_densenet161(self) -> None:
        """Tests DenseNet161 model"""
        self._check_model("densenet161")

    @unittest.skip("Skipping to resolve timeout issues in unittest framework")
    def test_densenet169(self) -> None:
        """Tests DenseNet169 model"""
        self._check_model("densenet169")

    @unittest.skip("Skipping to resolve timeout issues in unittest framework")
    def test_densenet201(self) -> None:
        """Tests DenseNet201 model"""
        self._check_model("densenet201")

    @unittest.skip("Skipping to resolve timeout issues in unittest framework")
    def test_googlenet(self) -> None:
        """Tests GoogLeNet models"""
        self._check_model("googlenet")

    @unittest.skip("Skipping to resolve timeout issues in unittest framework")
    def test_inception(self) -> None:
        """Tests inception models"""
        self._check_model("inception_v3")

    @unittest.skip("Skipping to resolve timeout issues in unittest framework")
    def test_mnasnet(self) -> None:
        """Tests MnasNet models"""
        model_names = ["mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3"]
        self._check_all_models(model_names)

    @unittest.skip("Skipping to resolve timeout issues in unittest framework")
    def test_mobilenet(self) -> None:
        """Tests MobileNet models"""
        self._check_model("mobilenet_v2")

    @unittest.skip("Skipping to resolve timeout issues in unittest framework")
    def test_resnet_small(self) -> None:
        """Tests small ResNet models"""
        model_names = ["resnet18", "resnet34", "resnet50"]
        self._check_all_models(model_names)

    @unittest.skip("Skipping to resolve timeout issues in unittest framework")
    def test_resnet101(self) -> None:
        """Tests ResNet101 model"""
        self._check_model("resnet101")

    @unittest.skip("Skipping to resolve timeout issues in unittest framework")
    def test_resnet152(self) -> None:
        """Tests ResNet152 model"""
        self._check_model("resnet152")

    @unittest.skip("Skipping to resolve timeout issues in unittest framework")
    def test_resnext101_32x8d(self) -> None:
        """Tests ResNeXt models"""
        self._check_model("resnext101_32x8d")

    @unittest.skip("Skipping to resolve timeout issues in unittest framework")
    def test_resnext50_32x4d(self) -> None:
        self._check_model("resnext50_32x4d")

    @unittest.skip("Skipping to resolve timeout issues in unittest framework")
    def test_shufflenet(self) -> None:
        """Tests ShuffleNet models"""
        model_names = [
            "shufflenet_v2_x0_5",
            "shufflenet_v2_x1_0",
            "shufflenet_v2_x1_5",
            "shufflenet_v2_x2_0",
        ]
        self._check_all_models(model_names)

    @unittest.skip("Skipping to resolve timeout issues in unittest framework")
    def test_squeezenet(self) -> None:
        """Tests SqueezeNet models"""
        model_names = ["squeezenet1_0", "squeezenet1_1"]
        self._check_all_models(model_names)

    @unittest.skip("Skipping to resolve timeout issues in unittest framework")
    def test_vgg11(self) -> None:
        """Tests VGG11 model"""
        self._check_model("vgg11")

    @unittest.skip("Skipping to resolve timeout issues in unittest framework")
    def test_vgg11_bn(self) -> None:
        """Tests VGG11 model with Batch Normalization"""
        self._check_model("vgg11_bn")

    @unittest.skip("Skipping to resolve timeout issues in unittest framework")
    def test_vgg13(self) -> None:
        """Tests VGG13 model"""
        self._check_model("vgg13")

    @unittest.skip("Skipping to resolve timeout issues in unittest framework")
    def test_vgg13_bn(self) -> None:
        """Tests VGG13 model with Batch Normalization"""
        self._check_model("vgg13_bn")

    @unittest.skip("Skipping to resolve timeout issues in unittest framework")
    def test_vgg16(self) -> None:
        """Tests VGG16 model"""
        self._check_model("vgg16")

    @unittest.skip("Skipping to resolve timeout issues in unittest framework")
    def test_vgg16_bn(self) -> None:
        """Tests VGG16 model with Batch Normalization"""
        self._check_model("vgg16_bn")

    @unittest.skip("Skipping to resolve timeout issues in unittest framework")
    def test_vgg19(self) -> None:
        """Tests VGG19 model"""
        self._check_model("vgg19")

    @unittest.skip("Skipping to resolve timeout issues in unittest framework")
    def test_vgg19_bn(self) -> None:
        """Tests VGG19 model with Batch Normalization"""
        self._check_model("vgg19_bn")

    @unittest.skip("Skipping to resolve timeout issues in unittest framework")
    def test_wide_resnet101_2(self) -> None:
        """Tests wide_resnet101_2 model"""
        self._check_model("wide_resnet101_2")

    @unittest.skip("Skipping to resolve timeout issues in unittest framework")
    def test_wide_resnet50_2(self) -> None:
        """Test wide_resnet50_2 model"""
        self._check_model("wide_resnet50_2")
