import sys
from unittest import TestCase

import torch
import torch.utils.data
import torchvision
from torch import optim, nn
from torchvision import transforms

from not_final_kernels.final_local_but_oom_kernel import BaseMLPClassifier, BaseMLPTrainer

MNIST_HIGHT_WIDTH = 28
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


def flatten(t):
    return t.view((t.size()[-1] * t.size()[-2]))


class TestBaseMLPClassifierAndTrainer(TestCase):

    # TODO assertion
    def test(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,)),
             transforms.Lambda(flatten)
             ]
        )

        dataset = torchvision.datasets.MNIST(root="../data/test",
                                             train=True,
                                             download=True,
                                             transform=transform)

        trainlaoader = torch.utils.data.DataLoader(dataset, batch_size=100,
                                                   shuffle=True, num_workers=4)

        testset = torchvision.datasets.MNIST(root="../data/test",
                                             train=False,
                                             download=True,
                                             transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                                 shuffle=False, num_workers=4)
        classes = list(range(10))

        model = BaseMLPClassifier(
            [{"in_features": MNIST_HIGHT_WIDTH * MNIST_HIGHT_WIDTH, "out_features": 256, "bias": True},
             {"in_features": 256, "out_features": 64, "bias": True},
             {"in_features": 64, "out_features": len(classes), "bias": True},
             ]
        )

        score_fuction = lambda predicted, labels: 100 * (predicted.argmax(dim=1) == labels).sum().item() / labels.size(
            0)
        print(list(model.parameters()))
        optimizer_factory = lambda model: optim.SGD(list(model.parameters()), lr=0.02, momentum=0.9)
        trainer = BaseMLPTrainer(model, loss_function=nn.CrossEntropyLoss(),
                                 score_function=score_fuction,
                                 optimizer_factory=optimizer_factory)

        trainer.train(trainlaoader, testloader, 40)

        # TODO assertion

    def test_early_stopping(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,)),
             transforms.Lambda(flatten)
             ]
        )

        dataset = torchvision.datasets.MNIST(root="../data/test",
                                             train=True,
                                             download=True,
                                             transform=transform)

        trainlaoader = torch.utils.data.DataLoader(dataset, batch_size=100,
                                                   shuffle=True, num_workers=4)

        testset = torchvision.datasets.MNIST(root="../data/test",
                                             train=False,
                                             download=True,
                                             transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                                 shuffle=False, num_workers=4)
        classes = list(range(10))

        model = BaseMLPClassifier(
            [{"in_features": MNIST_HIGHT_WIDTH * MNIST_HIGHT_WIDTH, "out_features": 256, "bias": True},
             {"in_features": 256, "out_features": 64, "bias": True},
             {"in_features": 64, "out_features": len(classes), "bias": True},
             ]
        )

        score_fuction = lambda predicted, labels: 100 * (predicted.argmax(dim=1) == labels).sum().item() / labels.size(
            0)
        print(list(model.parameters()))
        optimizer_factory = lambda model: optim.SGD(list(model.parameters()), lr=0.2, momentum=0.9)
        trainer = BaseMLPTrainer(model, loss_function=nn.CrossEntropyLoss(),
                                 score_function=score_fuction,
                                 optimizer_factory=optimizer_factory)

        trainer.train(trainlaoader, testloader, 40)
