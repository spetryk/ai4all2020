import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import nilearn
from nilearn import image


import os

"""
import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)
"""

class BrainomicsDataset(Dataset):
    def __init__(self, dataset_dir, train, transform=None):
        self.dataset_dir = dataset_dir
        self.train = train
        self.transform = transform

        # using slices 45 to 145 (they look the nicest)
        self.n_slices = 100

        # there are 94 subjects: sub-S01 - sub-S94
        self.n_train_subjects = 90
        self.n_val_subjects = 4

        n_subjects = self.n_train_subjects + self.n_val_subjects

        #self.data = torch.zeros(n_subjects * self.n_slices, 1, 240, 128)
        data = []
        for subject in range(1, n_subjects + 1):
            #subject = (idx // self.n_slices) + 1
            #data_slice = (idx % self.n_slices)# + 45

            subject_dir = "sub-S{:0>2d}/anat".format(subject)
            #fn = "sub-S{:0>2d}_T1w.nii.gz".format(subject)
            fn = "slices_all.npy"
            data_path = os.path.join(self.dataset_dir, subject_dir, fn)

            #mri_data = nilearn.image.load_img(data_path).get_data()
            mri_data = torch.load(data_path).unsqueeze(1)
            for i, image in enumerate(mri_data):
                transformed = self.transform(image)
                #print("transformed min/mean/max", transformed.min(), transformed.mean(), transformed.max())
                #self.data[(subject - 1) * self.n_slices + i,:,:,:] = transformed
                data.append(transformed)
        self.data = torch.stack(data)


    def __getitem__(self, idx):
        if not self.train:
            # 90 subjects in train, the rest in val
            idx += self.n_train_subjects * self.n_slices
        return self.data[idx]

    def __len__(self):
        # subjects 1 - 90 are train, 91 - 94 are val
        # (no need for val for gans anyway)
        if self.train:
            return self.n_train_subjects * self.n_slices
        else:
            return self.n_val_subjects * self.n_slices

def get_brainomics_dataloaders(base_dir, batch_size=64):
    all_transforms = transforms.Compose([
        transforms.Normalize(0, 32768),
        transforms.ToPILImage(),
        transforms.CenterCrop([240, 128]),
        transforms.ToTensor(),
        #transforms.Normalize(2635, 3561),
    ])

    #base_dir = "/work/drothchild/datasets/brainomics/localizer"
    train_data = BrainomicsDataset(base_dir, train=True,
                                   transform=all_transforms)
    test_data = BrainomicsDataset(base_dir, train=False,
                                  transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return train_loader, test_loader

def get_mnist_dataloaders(batch_size=128):
    """MNIST dataloader with (32, 32) sized images."""
    # Resize images so they are a power of 2
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    # Get train and test data
    train_data = datasets.MNIST('../data', train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.MNIST('../data', train=False,
                               transform=all_transforms)
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_fashion_mnist_dataloaders(batch_size=128):
    """Fashion MNIST dataloader with (32, 32) sized images."""
    # Resize images so they are a power of 2
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    # Get train and test data
    train_data = datasets.FashionMNIST('../fashion_data', train=True, download=True,
                                       transform=all_transforms)
    test_data = datasets.FashionMNIST('../fashion_data', train=False,
                                      transform=all_transforms)
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_lsun_dataloader(path_to_data='../lsun', dataset='bedroom_train',
                        batch_size=64):
    """LSUN dataloader with (128, 128) sized images.

    path_to_data : str
        One of 'bedroom_val' or 'bedroom_train'
    """
    # Compose transforms
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor()
    ])

    # Get dataset
    lsun_dset = datasets.LSUN(db_path=path_to_data, classes=[dataset],
                              transform=transform)

    # Create dataloader
    return DataLoader(lsun_dset, batch_size=batch_size, shuffle=True)

class Generator(nn.Module):
    def __init__(self, img_size, latent_dim, dim=16):
        super(Generator, self).__init__()

        self.dim = dim
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.feature_sizes = (self.img_size[0] // 16, self.img_size[1] // 16)

        self.latent_to_features = nn.Sequential(
            nn.Linear(latent_dim, 8 * dim * self.feature_sizes[0] * self.feature_sizes[1]),
            nn.ReLU()
        )

        self.features_to_image = nn.Sequential(
            nn.ConvTranspose2d(8 * dim, 4 * dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(4 * dim),
            nn.ConvTranspose2d(4 * dim, 2 * dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(2 * dim),
            nn.ConvTranspose2d(2 * dim, dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.ConvTranspose2d(dim, self.img_size[2], 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        # Map latent into appropriate size for transposed convolutions
        x = self.latent_to_features(input_data)
        # Reshape
        x = x.view(-1, 8 * self.dim, self.feature_sizes[0], self.feature_sizes[1])
        # Return generated image
        return self.features_to_image(x)

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))


class Discriminator(nn.Module):
    def __init__(self, img_size, dim=16):
        """
        img_size : (int, int, int)
            Height and width must be powers of 2.  E.g. (32, 32, 1) or
            (64, 128, 3). Last number indicates number of channels, e.g. 1 for
            grayscale or 3 for RGB
        """
        super(Discriminator, self).__init__()

        self.img_size = img_size

        self.image_to_features = nn.Sequential(
            nn.Conv2d(self.img_size[2], dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, 2 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * dim, 4 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4 * dim, 8 * dim, 4, 2, 1),
            nn.Sigmoid()
        )

        # 4 convolutions of stride 2, i.e. halving of size everytime
        # So output size will be 8 * (img_size / 2 ^ 4) * (img_size / 2 ^ 4)
        output_size = 8 * dim * (img_size[0] // 16) * (img_size[1] // 16)
        self.features_to_prob = nn.Sequential(
            nn.Linear(output_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        batch_size = input_data.size()[0]
        x = self.image_to_features(input_data)
        x = x.view(batch_size, -1)
        return self.features_to_prob(x)


def update(opt, loss):
    loss.backward()
    opt.step()

def gradient_penalty(discriminator, real_images, generated_images):
    batch_size = real_images.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_images).cuda()
    interpolated = alpha * real_images + (1 - alpha) * generated_images
    interpolated.requires_grad = True
    interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = discriminator(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)
    #self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return 10 * ((gradients_norm - 1) ** 2).mean()
