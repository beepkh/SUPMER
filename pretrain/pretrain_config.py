from .dataset.NSPPretrainDataset import NSPPretrainDataset
from .dataset.NSSPretrainDataset import NSSPretrainDataset
from .dataset.SingSentPretrainDataset import SingSentPretrainDataset

from .metatask.NextSentenceTask import NextSentenceTask
from .metatask.SameClusterSentSelectionTask import SameClusterSentSelectionTask
from .metatask.SameClusterSentPredTask import SameClusterSentPredTask
from .metatask.SentClusterClassifiTask import SentClusterClassifiTask

DATA_CONFIG = {
    "nss":{
        "dataset": NSSPretrainDataset
    }, 
    "nsp":{
        "dataset": NSPPretrainDataset
    }, 
    "ss":{
        "dataset": SingSentPretrainDataset
    }
}

TASK_CONFIG = {
    'scss': SameClusterSentSelectionTask,
    'ns': NextSentenceTask,
    'scsp': SameClusterSentPredTask,
    'scc': SentClusterClassifiTask
}