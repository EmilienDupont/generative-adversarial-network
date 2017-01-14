"""GAN that learns to generate samples from N(5,1) given N(0,1) noise"""
import numpy as np
import plotly.graph_objs as go
import plotly.offline as py

from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dense, Input
from keras.layers import Convolution1D
from keras.models import Model
from keras.optimizers import Adam
from plotly import tools

## Model parameters
data_dim = 10
noise_dim = 3

## Training parameters
batch_size = 32
num_epochs = 20
num_examples = 10000
num_batches = num_examples / batch_size

## Create generator
# Take as input noise and generate "fake" data
noise_input = Input(shape=(noise_dim,))
x = Dense(16, activation=LeakyReLU(0.1))(noise_input)
generator_output = Dense(output_dim=data_dim)(x)
generator = Model(input=noise_input, output=generator_output)
# Generator loss does not matter, as generator will only be trained through GAN
generator.compile(loss='mse', optimizer='adam')

## Create discriminator
# Take as input data or fake data and output probability of data being real
discriminator_input = Input(shape=(data_dim,))
x = Dense(16, activation=LeakyReLU(0.1))(discriminator_input)
x = Dense(16, activation=LeakyReLU(0.1))(x)
discriminator_output = Dense(1, activation='sigmoid')(x)
discriminator = Model(input=discriminator_input, output=discriminator_output)
discriminator.compile(loss='binary_crossentropy', optimizer='adam')

## Create Generative Adverserial Network
# The adverserial network is used to train the generator. Take as input noise
# and output probability of data being real. Maximize this.
discriminator.trainable = False # Freeze discriminator weights
gan_input = Input(shape=(noise_dim,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = Model(input=gan_input, output=gan_output)
gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=2e-4))

discriminator.trainable = True

## Inputs
data = np.random.normal(5, 1, (num_examples, data_dim))
def sample_noise(mean=0, std=1, shape=(batch_size, noise_dim)):
    return np.random.normal(mean, std, shape)

## Plotting
def plot(data, num_points=256):
    noise_batch = sample_noise(shape=(num_points, noise_dim))
    fake_data_batch = generator.predict(noise_batch)
    data_batch = data[np.random.randint(num_examples, size=num_points)]

    # Plot distributions
    trace_fake = go.Scatter(
        x = fake_data_batch[:,0],
        y = fake_data_batch[:,1],
        mode = 'markers',
        name='Generated Data'
    )

    trace_real = go.Scatter(
        x = data_batch[:,0],
        y = data_batch[:,1],
        mode = 'markers',
        name = 'Real Data'
    )

    data = [trace_fake, trace_real]
    fig = go.Figure(data=data)
    py.plot(fig)

def train_discriminator(data_batch=None):
    """Train D on real or fake data."""
    if data_batch is None: # Train on fake data
        noise_batch = sample_noise()
        fake_data_batch = generator.predict(noise_batch)
        labels = np.zeros(batch_size).astype(int)
        discriminator_loss = \
                discriminator.train_on_batch(fake_data_batch, labels)
    else: # Train on real data
        labels = np.ones(batch_size).astype(int)
        discriminator_loss = \
                discriminator.train_on_batch(data_batch, labels)
    return discriminator_loss

def train_generator():
    """Train the generator."""
    noise_batch = sample_noise()
    # Want to fool discriminator, i.e. make D output 1
    labels = np.ones(batch_size).astype(int)
    gan_loss = gan.train_on_batch(noise_batch, labels)
    return gan_loss

# Pretrain discriminator
shuffled_data = np.random.permutation(data)
for batch in range(num_batches):
    data_batch = shuffled_data[batch * batch_size: (batch + 1) * batch_size]
    train_discriminator()
    train_discriminator(data_batch)

## Training
for epoch in range(num_epochs):

    if epoch % 2 == 0: plot(data)
    print("\nEpoch {}".format(epoch + 1))
    shuffled_data = np.random.permutation(data)

    for batch in range(num_batches):

        data_batch = shuffled_data[batch * batch_size: (batch + 1) * batch_size]
        discriminator_loss = train_discriminator()
        discriminator_loss = train_discriminator(data_batch)
        gan_loss = train_generator()
        gan_loss = train_generator()

        if batch % 50 == 0:
            print "batch", batch
            print "\t", "gan_loss:", gan_loss, "discriminator_loss", discriminator_loss
