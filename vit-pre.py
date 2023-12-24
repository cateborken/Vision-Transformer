import matplotlib.pyplot as plt
import torch
import torchvision
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from going_modular.going_modular import engine

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # Setup directory paths to train and test images
    train_dir = 'D:/Tools/AI Training datasets/Custom/train'
    test_dir = 'D:/Tools/AI Training datasets/Custom/test'

    NUM_WORKERS = 4


    def create_dataloaders(
            train_dir: str,
            test_dir: str,
            transform: transforms.Compose,
            batch_size: int,
            num_workers: int = NUM_WORKERS
    ):
        # Use ImageFolder to create dataset(s)
        train_data = datasets.ImageFolder(train_dir, transform=transform)
        test_data = datasets.ImageFolder(test_dir, transform=transform)

        # Get class names
        class_names = train_data.classes

        # Turn images into data loaders
        train_dataloader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_dataloader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        return train_dataloader, test_dataloader, class_names


    # Create image size
    IMG_SIZE = 224

    # Create transform pipeline manually
    manual_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ])

    # Set the batch size
    BATCH_SIZE = 32

    # Create data loaders
    train_dataloader, test_dataloader, class_names = create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=manual_transforms,
        batch_size=BATCH_SIZE
    )

    # Get a batch of images
    image_batch, label_batch = next(iter(train_dataloader))
    # Get a single image from the batch
    image, label = image_batch[0], label_batch[0]


    # 1. Create a class which subclasses nn.Module
    class PatchEmbedding(nn.Module):
        """Turns a 2D input image into a 1D sequence learnable embedding vector.

        Args:
            in_channels (int): Number of color channels for the input images. Defaults to 3.
            patch_size (int): Size of patches to convert input image into. Defaults to 16.
            embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
        """

        # 2. Initialize the class with appropriate variables
        def __init__(self,
                     in_channels: int = 3,
                     patch_size: int = 16,
                     embedding_dim: int = 768):
            super().__init__()

            # 3. Create a layer to turn an image into patches
            self.patcher = nn.Conv2d(in_channels=in_channels,
                                     out_channels=embedding_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size,
                                     padding=0)

            # 4. Create a layer to flatten the patch feature maps into a single dimension
            self.flatten = nn.Flatten(start_dim=2,  # only flatten the feature map dimensions into a single vector
                                      end_dim=3)

        # 5. Define the forward method
        def forward(self, x):
            # Create assertion to check that inputs are the correct shape
            image_resolution = x.shape[-1]
            assert image_resolution % patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {patch_size}"

            # Perform the forward pass
            x_patched = self.patcher(x)
            x_flattened = self.flatten(x_patched)

            # 6. Make sure the output shape has the right order
            return x_flattened.permute(0, 2, 1)


    # Patch
    patch_size = 16


    # Set seeds
    def set_seeds(seed: int = 42):
        """Sets random sets for torch operations.

        Args:
            seed (int, optional): Random seed to set. Defaults to 42.
        """
        # Set the seed for general torch operations
        torch.manual_seed(seed)
        # Set the seed for CUDA torch operations (ones that happen on the GPU)
        torch.cuda.manual_seed(seed)


    set_seeds()

    # 2. Get the image dimensions
    height, width = image.shape[1], image.shape[2]

    # 3. Get image tensor and add batch dimension
    x = image.unsqueeze(0)

    # 4. Create patch embedding layer
    patch_embedding_layer = PatchEmbedding(in_channels=3,
                                           patch_size=patch_size,
                                           embedding_dim=768)

    # 5. Pass image through patch embedding layer
    patch_embedding = patch_embedding_layer(x)

    # 6. Create class token embedding
    batch_size = patch_embedding.shape[0]
    embedding_dimension = patch_embedding.shape[-1]
    class_token = nn.Parameter(torch.ones(batch_size, 1, embedding_dimension),
                               requires_grad=True)

    # 7. Prepend class token embedding to patch embedding
    patch_embedding_class_token = torch.cat((class_token, patch_embedding), dim=1)

    # 8. Create position embedding
    number_of_patches = int((height * width) / patch_size ** 2)
    position_embedding = nn.Parameter(torch.ones(1, number_of_patches + 1, embedding_dimension),
                                      requires_grad=True)

    # 9. Add position embedding to patch embedding with class token
    patch_and_position_embedding = patch_embedding_class_token + position_embedding


    # 10. Create a class that inherits from nn.Module
    class MultiheadSelfAttentionBlock(nn.Module):
        """Creates a multi-head self-attention block ("MSA block" for short).
        """

        # 11. Initialize the class with hyperparameters from Table 1
        def __init__(self,
                     embedding_dim: int = 768,  # Hidden size D from Table 1 for ViT-Base
                     num_heads: int = 12,  # Heads from Table 1 for ViT-Base
                     attn_dropout: float = 0):  # doesn't look like the paper uses any dropout in MSABlocks
            super().__init__()

            # 12. Create the Norm layer (LN)
            self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

            # 13. Create the Multi-Head Attention (MSA) layer
            self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                        num_heads=num_heads,
                                                        dropout=attn_dropout,
                                                        batch_first=True)  # does our batch dimension come first?

        # 14. Create a forward() method to pass the data throguh the layers
        def forward(self, x):
            x = self.layer_norm(x)
            attn_output, _ = self.multihead_attn(query=x,  # query embeddings
                                                 key=x,  # key embeddings
                                                 value=x,  # value embeddings
                                                 need_weights=False)  # do we need the weights or just the layer outputs?
            return attn_output


    # 15. Create a class that inherits from nn.Module
    class MLPBlock(nn.Module):
        """Creates a layer normalized multilayer perceptron block ("MLP block" for short)."""

        # 16. Initialize the class with hyperparameters from Table 1 and Table 3
        def __init__(self,
                     embedding_dim: int = 768,  # Hidden Size D from Table 1 for ViT-Base
                     mlp_size: int = 3072,  # MLP size from Table 1 for ViT-Base
                     dropout: float = 0.1):  # Dropout from Table 3 for ViT-Base
            super().__init__()

            # 17. Create the Norm layer (LN)
            self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

            # 18. Create the Multilayer perceptron (MLP) layer(s)
            self.mlp = nn.Sequential(
                nn.Linear(in_features=embedding_dim,
                          out_features=mlp_size),
                nn.GELU(),  # "The MLP contains two layers with a GELU non-linearity (section 3.1)."
                nn.Dropout(p=dropout),
                nn.Linear(in_features=mlp_size,  # needs to take same in_features as out_features of layer above
                          out_features=embedding_dim),  # take back to embedding_dim
                nn.Dropout(p=dropout)  # "Dropout, when used, is applied after every dense layer.."
            )

        # 19. Create a forward() method to pass the data throguh the layers
        def forward(self, x):
            x = self.layer_norm(x)
            x = self.mlp(x)
            return x


    # 20. Create a class that inherits from nn.Module
    class TransformerEncoderBlock(nn.Module):
        """Creates a Transformer Encoder block."""

        # 21. Initialize the class with hyperparameters from Table 1 and Table 3
        def __init__(self,
                     embedding_dim: int = 768,  # Hidden size D from Table 1 for ViT-Base
                     num_heads: int = 12,  # Heads from Table 1 for ViT-Base
                     mlp_size: int = 3072,  # MLP size from Table 1 for ViT-Base
                     mlp_dropout: float = 0.1,  # Amount of dropout for dense layers from Table 3 for ViT-Base
                     attn_dropout: float = 0):  # Amount of dropout for attention layers
            super().__init__()

            # 22. Create MSA block (equation 2)
            self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                         num_heads=num_heads,
                                                         attn_dropout=attn_dropout)

            # 23. Create MLP block (equation 3)
            self.mlp_block = MLPBlock(embedding_dim=embedding_dim,
                                      mlp_size=mlp_size,
                                      dropout=mlp_dropout)

        # 24. Create a forward() method
        def forward(self, x):
            # 25. Create residual connection for MSA block (add the input to the output)
            x = self.msa_block(x) + x

            # 26. Create residual connection for MLP block (add the input to the output)
            x = self.mlp_block(x) + x

            return x


    # Vision Transformer
    # 1. Create a ViT class that inherits from nn.Module
    class ViT(nn.Module):
        """Creates a Vision Transformer architecture with ViT-Base hyperparameters by default."""

        # 2. Initialize the class with hyperparameters from Table 1 and Table 3
        def __init__(self,
                     img_size: int = 224,  # Training resolution from Table 3 in ViT paper
                     in_channels: int = 3,  # Number of channels in input image
                     patch_size: int = 16,  # Patch size
                     num_transformer_layers: int = 12,  # Layers from Table 1 for ViT-Base
                     embedding_dim: int = 768,  # Hidden size D from Table 1 for ViT-Base
                     mlp_size: int = 3072,  # MLP size from Table 1 for ViT-Base
                     num_heads: int = 12,  # Heads from Table 1 for ViT-Base
                     attn_dropout: float = 0,  # Dropout for attention projection
                     mlp_dropout: float = 0.1,  # Dropout for dense/MLP layers
                     embedding_dropout: float = 0.1,  # Dropout for patch and position embeddings
                     num_classes: int = 5):  # Default for ImageNet but can customize this
            super().__init__()  # don't forget the super().__init__()!

            # 3. Make the image size is divisble by the patch size
            assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}."

            # 4. Calculate number of patches (height * width/patch^2)
            self.num_patches = (img_size * img_size) // patch_size ** 2

            # 5. Create learnable class embedding (needs to go at front of sequence of patch embeddings)
            self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                                requires_grad=True)

            # 6. Create learnable position embedding
            self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches + 1, embedding_dim),
                                                   requires_grad=True)

            # 7. Create embedding dropout value
            self.embedding_dropout = nn.Dropout(p=embedding_dropout)

            # 8. Create patch embedding layer
            self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                                  patch_size=patch_size,
                                                  embedding_dim=embedding_dim)

            # 9. Create Transformer Encoder blocks (we can stack Transformer Encoder blocks using nn.Sequential())
            # Note: The "*" means "all"
            self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                               num_heads=num_heads,
                                                                               mlp_size=mlp_size,
                                                                               mlp_dropout=mlp_dropout) for _ in
                                                       range(num_transformer_layers)])

            # 10. Create classifier head
            self.classifier = nn.Sequential(
                nn.LayerNorm(normalized_shape=embedding_dim),
                nn.Linear(in_features=embedding_dim,
                          out_features=num_classes)
            )

        # 11. Create a forward() method
        def forward(self, x):
            # 12. Get batch size
            batch_size = x.shape[0]

            # 13. Create class token embedding and expand it to match the batch size (equation 1)
            class_token = self.class_embedding.expand(batch_size, -1,
                                                      -1)  # "-1" means to infer the dimension (try this line on its own)

            # 14. Create patch embedding (equation 1)
            x = self.patch_embedding(x)

            # 15. Concat class embedding and patch embedding (equation 1)
            x = torch.cat((class_token, x), dim=1)

            # 16. Add position embedding to patch embedding (equation 1)
            x = self.position_embedding + x

            # 17. Run embedding dropout (Appendix B.1)
            x = self.embedding_dropout(x)

            # 18. Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
            x = self.transformer_encoder(x)

            # 19. Put 0 index logit through classifier (equation 4)
            x = self.classifier(x[:, 0])  # run on each sample in a batch at 0 index

            return x


    # Create an instance of ViT with the number of classes we're working with
    vit = ViT(num_classes=len(class_names))


    # Setup the optimizer to optimize our ViT model parameters using hyperparameters from the ViT paper
    optimizer = torch.optim.Adam(params=vit.parameters(),
                                 lr=3e-3,  # Base LR from Table 3 for ViT-* ImageNet-1k
                                 betas=(0.9, 0.999),
                                 # default values but also mentioned in ViT paper section 4.1 (Training & Fine-tuning)
                                 weight_decay=0.3)  # from the ViT paper section 4.1 (Training & Fine-tuning) and Table 3 for ViT-* ImageNet-1k

    # Setup the loss function for multi-class classification
    loss_fn = torch.nn.CrossEntropyLoss()

    # Set the seeds
    set_seeds()

    # Train the model and save the training results to a dictionary
    results = engine.train(model=vit,
                           train_dataloader=train_dataloader,
                           test_dataloader=test_dataloader,
                           optimizer=optimizer,
                           loss_fn=loss_fn,
                           epochs=10,
                           device=device)

    # To check out our ViT model's loss curves, we can use the plot_loss_curves function from helper_functions.py
    from helper_functions import plot_loss_curves

    # Plot our ViT model's loss curves
    plot_loss_curves(results)