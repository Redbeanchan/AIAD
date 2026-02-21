import os
import pickle
import numpy as np

class DummySegmenter:
    def predict(self, img_arr):
        # img_arr: H,W,3
        gray = np.mean(img_arr, axis=2)
        mask = (gray > gray.mean() * 0.9).astype('uint8') * 255
        return mask

os.makedirs('models', exist_ok=True)
with open('models/segmenter.pkl', 'wb') as f:
    pickle.dump(DummySegmenter(), f)

print('Wrote models/segmenter.pkl (DummySegmenter)')
