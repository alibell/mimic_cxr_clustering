from torch.utils.data import Dataset, DataLoader
from collections.abc import Iterable
import numpy as np
from PIL import Image

class imageDataset(Dataset):
    def __init__ (self, images_paths):
        """
            Parameters:
            ----------
            images_paths : dict{int:str} containing image id and image path
        """
        super().__init__()

        self.image_paths = images_paths
        self.image_list = list(self.image_paths.keys()) # Useful for dataset splitting

    def split(self, uid_list=None, p=0.3, random_seed=None):
        """
            Split the dataset

            Parameters:
            -----------
            p : pourcentage of split repartition
            random_seed : int, random seed

            Output:
            -------
            Tuple of image dataset object
        """

        if isinstance(p, float) == False or p > 1 or p < 0:
            raise ValueError("p should be between 0 and 1")

        if random_seed is not None and isinstance(random_seed, int):
            np.random.RandomState(random_seed)

        if uid_list is None:
            # Getting the splitted image list
            n_items = len(self.image_list)
            n_1 = int(n_items*p)
            n_2 = n_items-n_1

            mask = np.zeros(n_items).astype("bool")
            mask[np.random.choice(range(n_items), replace=False, size=n_1)] = True
        else:
            # Getting the splitted image list
            mask = np.array([True if x in uid_list else False for x in self.image_paths.keys()])

        keys_1 = np.array(self.image_list)[mask]
        values_1 = np.array(list(self.image_paths.values()))[mask]
        dataset_1 = imageDataset(dict(zip(keys_1, values_1)))

        keys_2 = np.array(self.image_list)[mask == False]
        values_2 = np.array(list(self.image_paths.values()))[mask == False]
        dataset_2 = imageDataset(dict(zip(keys_2, values_2)))

        return dataset_1, dataset_2

    def _slice_to_list (self, s):

        # Getting indices range
        if isinstance(s, slice):
            start = s.start if s.start is not None else 0
            stop = s.stop if s.stop is not None else self.__len__()
            step = s.step if s.step is not None else 1
    
            indices = range(start, stop, step)
        elif isinstance(s, int):
            indices = [s]
        elif isinstance(s, Iterable):
            indices = s
        else:
            raise NotImplementedError

        return indices

    def __str__(self):
        n_data = self.__len__()

        return(f"CXR Dataset of {n_data} images")

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.image_paths.keys())

    def __get_image (self, image_path):
        image = np.array(
                    Image.open(image_path)
        )

        return image

    def __getitem__(self, s):
        images = []
        indices = self._slice_to_list(s)
        
        for idx in indices:
            images.append(
                self.__get_image(self.image_paths[self.image_list[idx]])
            )
        
        return images

    def get_from_id(self,  s):
        images = []
        indices = self._slice_to_list(s)

        for idx in indices:
            images.append(
                self.__get_image(self.image_paths[idx])
            )

        return images