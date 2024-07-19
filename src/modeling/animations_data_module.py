import pytorch_lightning as pl
from torch import Generator
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

class AnimationTypesDatamodule(pl.LightningDataModule):
  def __init__(self, batch_size = 32):
    super().__init__()
    self.batch_size = batch_size
  def setup(self, stage = None):
    transform = transforms.Compose([
                                    transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0), (1))
                                ])


    dataset = ImageFolder(root='animation_types/animation-types', transform=transform)

    seed = Generator().manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    self.train_dataset, self.test_dataset, self.val_dataset = random_split(
        dataset, [train_size, test_size, val_size], seed
    )


  def train_dataloader(self):
    return  DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True)
  def test_dataloader(self):
    return  DataLoader(self.test_dataset, batch_size = self.batch_size, shuffle = False)
  def val_dataloader(self):
    return  DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle = False)
