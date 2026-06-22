"""Architecture for autoencoder."""

import flax.linen as nn
from jax.image import resize

# Define the Neural Network
# ===============================================================================
class ConvDownBlock(nn.Module):
    """Convolutional down layer."""

    num_features: int
    mom: float
    kernel_size: int
    pooling_size: int
    leaky_val: float
    mom: float
    stride : int

    def setup(self):
        """Set up the layer."""
        self.Conv1 = nn.Conv(features=self.num_features,
                             kernel_size=(self.kernel_size,self.kernel_size),
                             strides=(self.stride,self.stride), padding='same')
        self.BN1 = nn.BatchNorm(momentum=self.mom)
        self.Conv2 = nn.Conv(features=self.num_features,
                             kernel_size=(self.kernel_size,self.kernel_size),
                             strides=(self.stride,self.stride), padding='same')
        self.BN2 = nn.BatchNorm(momentum=self.mom)


    def __call__(self, x, train: bool):
        """Call the layer."""
        x1 = self.Conv1(x)
        x2 = self.BN1(x1, use_running_average=not train)
        x3 = nn.leaky_relu(x2, self.leaky_val)
        x4 = self.Conv2(x3)
        x5 = self.BN2(x4, use_running_average=not train)
        x6 = nn.leaky_relu(x5, self.leaky_val)
        x7 = nn.max_pool(x6, window_shape=(self.pooling_size, self.pooling_size),
                         strides=(self.pooling_size, self.pooling_size))
        return x7

class ConvBlock(nn.Module):
    """Convolutional layer."""

    num_features: int
    mom: float
    kernel_size: int
    leaky_val: float
    stride: int

    def setup(self):
        """Set up the layer."""
        self.Conv1 = nn.Conv(features=self.num_features,
                             kernel_size=(self.kernel_size,self.kernel_size),
                             strides=(self.stride,self.stride), padding='same')
        self.BN1 = nn.BatchNorm(momentum=self.mom)
        self.Conv2 = nn.Conv(features=self.num_features,
                             kernel_size=(self.kernel_size,self.kernel_size),
                             strides=(self.stride,self.stride), padding='same')
        self.BN2 = nn.BatchNorm(momentum=self.mom)


    def __call__(self, x, train: bool):
        """Call the layer."""
        x1 = self.Conv1(x)
        x2 = self.BN1(x1, use_running_average=not train)
        x3 = nn.leaky_relu(x2, self.leaky_val)
        x4 = self.Conv2(x3)
        x5 = self.BN2(x4, use_running_average=not train)
        x6 = nn.leaky_relu(x5, self.leaky_val)
        return x6

class ConvUpBlock(nn.Module):
    """Convolutional up layer."""

    num_features: int
    mom: float
    kernel_size: int
    upsample_size: int
    leaky_val: float
    stride: int

    def setup(self):
        """Set up the layer."""
        self.Conv1 = nn.Conv(features=self.num_features,
                             kernel_size = (self.kernel_size,self.kernel_size),
                             strides=(self.stride,self.stride), padding='same')
        self.BN1 = nn.BatchNorm(momentum=self.mom)
        self.Conv2 = nn.Conv(features=self.num_features,
                             kernel_size=(self.kernel_size,self.kernel_size),
                             strides=(self.stride,self.stride), padding='same')
        self.BN2 = nn.BatchNorm(momentum=self.mom)

    def __call__(self, x, train: bool):
        """Call the layer."""
        batch_dim, x_dim, y_dim, f_dim = x.shape
        x1 = resize(x, shape=(batch_dim, self.upsample_size*x_dim,
                              self.upsample_size*y_dim, f_dim), method='nearest')
        x2 = self.Conv1(x1)
        x3 = self.BN1(x2, use_running_average=not train)
        x4 = nn.leaky_relu(x3, self.leaky_val)
        x5 = self.Conv2(x4)
        x6 = self.BN2(x5, use_running_average=not train)
        x7 = nn.leaky_relu(x6, self.leaky_val)
        return x7

class Autoencoder(nn.Module):
    """Autoencoder architecture."""

    num_down_blocks: int
    num_up_blocks: int
    num_base_filters: int
    kernel_size: int
    pooling_size: int
    upsample_size: int
    mom: float
    leaky_val: float
    stride: int
    out_layers: int
    out_size: int

    def setup(self):
        """Set up the architecture."""
        self.conv_downs=[ConvDownBlock(2**(i)*self.num_base_filters,
                                      self.mom, self.kernel_size, self.pooling_size,
                                      self.leaky_val, self.stride) 
                                      for i in range(self.num_down_blocks)]
        self.conv_block=ConvBlock(2**(self.num_down_blocks)*self.num_base_filters,
                                  self.mom, self.kernel_size,
                                  self.leaky_val, self.stride)
        self.conv_ups=[ConvUpBlock(2**(self.num_down_blocks-(i+1))*self.num_base_filters,
                                  self.mom, self.kernel_size, self.upsample_size,
                                  self.leaky_val, self.stride) 
                                  for i in range(self.num_up_blocks)]
        self.conv1=nn.Conv(features=self.out_layers,
                           kernel_size = (self.kernel_size,self.kernel_size),
                             strides=(self.stride,self.stride), padding='same')
        self.conv2=nn.Conv(features=self.out_layers,
                           kernel_size = (self.kernel_size,self.kernel_size),
                             strides=(self.stride,self.stride), padding='same')

    def __call__(self, x, training: bool):
        """Call the architecture."""
        for conv_down in self.conv_downs:
            x = conv_down(x, train=training)
        x = self.conv_block(x, train=training)
        for conv_up in self.conv_ups:
            x = conv_up(x, train=training)
        x = self.conv1(x)
        x = nn.leaky_relu(x, self.leaky_val)
        x = self.conv2(x)
        x = nn.relu(x)
        _, x_dist, y_dist, _ = x.shape
        #### Manually input size
        s = self.out_size//2
        x = x[:,x_dist//2-s:x_dist//2+s,y_dist//2-s:y_dist//2+s,:]
        return x