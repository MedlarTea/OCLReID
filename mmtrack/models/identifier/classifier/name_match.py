from .global_ocl_classifier import GlobalOCLClassifier
from .part_ocl_weighted_classifier import PartOCLWeightedClassifier
from .global_r18_naive_classifier import GlobalResNetClassifier
classifiers = {
    "GlobalOCLClassifier": GlobalOCLClassifier,
    "PartOCLWeightedClassifier": PartOCLWeightedClassifier,
    "GlobalResNetClassifier": GlobalResNetClassifier,
}