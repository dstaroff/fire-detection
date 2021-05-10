import os

from cv2 import cv2


class Model:
    def __init__(self):
        self.net = self._get_network()
        self.labels = self._get_labels()

        layer_names = self.net.getLayerNames()
        self.outputs = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def process_frame(self, frame):
        blob = cv2.dnn.blobFromImage(
            image=frame,
            scalefactor=1 / 255,
            size=(416, 416),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)

        return self.net.forward(self.outputs)

    @staticmethod
    def _get_network():
        return cv2.dnn.readNet(
            model=os.path.join(os.path.dirname(__file__), 'data', 'model.weights'),
            config=os.path.join(os.path.dirname(__file__), 'data', 'model.cfg'),
        )

    @staticmethod
    def _get_labels():
        with open(os.path.join(os.path.dirname(__file__), 'data', 'labels.txt'), mode='r') as labels_file:
            return [line.strip() for line in labels_file.readlines()]
