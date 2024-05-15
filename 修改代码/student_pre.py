import torch




class Collate():
    def __init__(self, n_degrades) -> None:
        self.n_degrades = n_degrades

    def __call__(self, batch):

        target_images = [[] for _ in range(self.n_degrades)]
        input_images = [[] for _ in range(self.n_degrades)]

        for i in range(len(batch)):
            target_image, input_image, dataset_label = batch[i]
            target_images[dataset_label].append(target_image.unsqueeze(0))
            input_images[dataset_label].append(input_image.unsqueeze(0))

        for i in range(len(target_images)):
            if target_images[i] == []:
                return None, None
            target_images[i] = torch.cat(target_images[i])
            input_images[i] = torch.cat(input_images[i])
        target_images = torch.cat(target_images)

        return target_images, input_images