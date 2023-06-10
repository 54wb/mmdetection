from .builder import DATASETS
from .coco import CocoDataset

@DATASETS.register_module()
class YaoDataset(CocoDataset):

    CLASSES = ('office', 'otherRoom', 'toilet', 'openAera', 'stair', 
        'lift', 'hotel', 'dormitory', 'classroom', 'ward')

    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30)]
