from train import data_process
from model import Generator, Discriminator, DiscriminatorForWGAN
import torch
from train import train_GAN, train_WGAN, train_WGAN_GP
from image_generate import gif_generate

if __name__ == '__main__':
    task = 'gif_generate'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    input_size = 2
    epochs = 2000
    data_process(batch_size, input_size)
    if task == 'GAN_SGD':
        test_, train_dataloader, test_dataloader = data_process(batch_size=batch_size, input_size=input_size)
        generator = Generator(input_size, 2).to(device)
        discriminator = Discriminator(2, 1).to(device)
        generator_optimizer = torch.optim.SGD(generator.parameters(), lr=0.0005)
        discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=0.0005)
        train_GAN(epochs=epochs,
                  train_dataloader=train_dataloader,
                  generator=generator,
                  discriminator=discriminator,
                  generator_optimizer=generator_optimizer,
                  discriminator_optimizer=discriminator_optimizer,
                  input_size=input_size,
                  test_data=test_,
                  task=task
                  )

    if task == 'GAN_Adam':
        test_, train_dataloader, test_dataloader = data_process(batch_size=batch_size, input_size=input_size)
        generator = Generator(input_size, 2).to(device)
        discriminator = Discriminator(2, 1).to(device)
        generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.00005)
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.00005)
        train_GAN(epochs=epochs,
                  train_dataloader=train_dataloader,
                  generator=generator,
                  discriminator=discriminator,
                  generator_optimizer=generator_optimizer,
                  discriminator_optimizer=discriminator_optimizer,
                  input_size=input_size,
                  test_data=test_,
                  task=task
                  )

    if task == 'WGAN_SGD':
        test_, train_dataloader, test_dataloader = data_process(batch_size=batch_size, input_size=input_size)
        generator = Generator(input_size, 2).to(device)
        discriminator = DiscriminatorForWGAN(2, 1).to(device)
        generator_optimizer = torch.optim.SGD(generator.parameters(), lr=0.0005)
        discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=0.0005)
        train_WGAN(epochs=epochs,
                   train_dataloader=train_dataloader,
                   generator=generator,
                   discriminator=discriminator,
                   generator_optimizer=generator_optimizer,
                   discriminator_optimizer=discriminator_optimizer,
                   input_size=input_size,
                   test_data=test_,
                   clamp=0.1,
                   task=task
                   )

    if task == 'WGAN_Adam':
        test_, train_dataloader, test_dataloader = data_process(batch_size=batch_size, input_size=input_size)
        generator = Generator(input_size, 2).to(device)
        discriminator = DiscriminatorForWGAN(2, 1).to(device)
        generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.000003)
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.000003)
        train_WGAN(epochs=epochs,
                   train_dataloader=train_dataloader,
                   generator=generator,
                   discriminator=discriminator,
                   generator_optimizer=generator_optimizer,
                   discriminator_optimizer=discriminator_optimizer,
                   input_size=input_size,
                   test_data=test_,
                   clamp=0.1,
                   task=task
                  )

    if task == 'WGAN_GP_SGD':
        test_, train_dataloader, test_dataloader = data_process(batch_size=batch_size, input_size=input_size)
        generator = Generator(input_size, 2).to(device)
        discriminator = DiscriminatorForWGAN(2, 1).to(device)
        generator_optimizer = torch.optim.SGD(generator.parameters(), lr=0.0005)
        discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=0.0005)
        train_WGAN_GP(epochs=epochs,
                   train_dataloader=train_dataloader,
                   generator=generator,
                   discriminator=discriminator,
                   generator_optimizer=generator_optimizer,
                   discriminator_optimizer=discriminator_optimizer,
                   input_size=input_size,
                   test_data=test_,
                   clamp=0.1,
                   task=task
                   )

    if task == 'WGAN_GP_Adam':
        test_, train_dataloader, test_dataloader = data_process(batch_size=batch_size, input_size=input_size)
        generator = Generator(input_size, 2).to(device)
        discriminator = DiscriminatorForWGAN(2, 1).to(device)
        generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.000001)
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.000001)
        train_WGAN(epochs=epochs,
                   train_dataloader=train_dataloader,
                   generator=generator,
                   discriminator=discriminator,
                   generator_optimizer=generator_optimizer,
                   discriminator_optimizer=discriminator_optimizer,
                   input_size=input_size,
                   test_data=test_,
                   clamp=0.1,
                   task=task
                   )

    if task == 'gif_generate':
        gif_generate('./result/GAN_SGD')
        gif_generate('./result/GAN_Adam')
        gif_generate('./result/WGAN_SGD')
        gif_generate('./result/WGAN_Adam')
        gif_generate('./result/WGAN_GP_SGD')
        gif_generate('./result/WGAN_GP_Adam')




