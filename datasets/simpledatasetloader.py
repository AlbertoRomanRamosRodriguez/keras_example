import numpy as np
import cv2
from os.path import sep

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None) -> None:
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []
    
    def load(self, imagePaths, verbose=-1):
        data = []
        labels = []

        for (i, imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            label= imagePath.split(sep)[-2]

            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print(f"[INFO] processing {imagePath.split(sep)[-1]} {i+1}/{len(imagePaths)}")
            
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)
            
            data.append(image)
            labels.append(label)

            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print(f"[INFO] processed {imagePath.split(sep)[-1]}")

        return (np.array(data), np.array(labels))
