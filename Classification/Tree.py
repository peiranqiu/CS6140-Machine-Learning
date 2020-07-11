class MultiSplitTree:
    def __init__(self):
        self.entropy = None
        self.mostLabel = None
        self.attribute = None
        self.value = None
        self.children = []
        self.isLeaf = False
        self.label = None

class BinarySplitTree:
    def __init__(self):
        self.entropy = None
        self.mostLabel = None
        self.attribute = None
        self.value = None
        self.leftChild = None
        self.rightChild = None
        self.isLeaf = False
        self.label = None