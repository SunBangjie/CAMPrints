import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, output_size=1024, input_size=(3, 1024, 1024)):
        super(Encoder, self).__init__()

        self.input_size = input_size

        # Convolution layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(16, 32, 4, 2, 1, dilation=2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, 4, 2, 1, dilation=2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, dilation=2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 128, 4, 2, 1, dilation=2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 3, 2, 1, dilation=2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Dynamically calculate the size of the output from the convolutional layers
        self.conv_output_size = self._get_conv_output_size(input_size)
        
        self.linear = nn.Sequential(
            nn.Linear(self.conv_output_size, output_size),
            nn.LeakyReLU(0.2),
        )
    
    def _get_conv_output_size(self, shape):
        # Create a dummy input tensor with the given shape
        dummy_input = torch.zeros(1, *shape)
        # Pass the dummy input through the convolutional layers
        output = self.conv(dummy_input)
        # Calculate the size of the output tensor
        return int(torch.prod(torch.tensor(output.size()[1:])))

    def forward(self, x):
        x = x.view(-1, *self.input_size)
        x = self.conv(x)
        x = x.view(-1, self.conv_output_size)
        return self.linear(x)

