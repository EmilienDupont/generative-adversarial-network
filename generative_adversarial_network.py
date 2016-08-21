import numpy as np

from keras.layers import Dense, Input
from keras.layers import Convolution1D
from keras.models import Model

## Model parameters
data_dim = 128
noise_dim = 64

## Training parameters
batch_size = 32
num_steps = 1000
num_examples = 10000

## Create generator
# Take as input noise and generate "fake" data
noise_input = Input(shape=(noise_dim,))
x = Dense(128, activation='relu')(noise_input)
x = Dense(256, activation='relu')(x)
generator_output = Dense(output_dim=data_dim)(x)
generator = Model(input=noise_input, output=generator_output)
generator.compile(loss='mse', optimizer='adam') # We need to compile it to be able to use it?

## Create discriminator
# Take as input data or fake data and output probability of data being real
discriminator_input = Input(shape=(data_dim,))
x = Dense(128)(discriminator_input)
discriminator_output = Dense(1, activation='sigmoid')(x) # This outputs a probability
discriminator = Model(input=discriminator_input, output=discriminator_output)
discriminator.compile(loss='binary_crossentropy', optimizer='adam')

discriminator.trainable = False # Freeze discriminator weights

## Create Generative Adverserial Network
# The adverserial network is used to train the generator. Take as input noise
# and output probability of data being real. Maximize this.
gan_input = Input(shape=(noise_dim,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = Model(input=gan_input, output=gan_output)
gan.compile(loss='binary_crossentropy', optimizer='adam')

discriminator.trainable = True

## Inputs
data = np.random.normal(0, 1, (num_examples, data_dim))

## Training
for step in range(num_steps):
    # Train the generator
    noise = np.random.uniform(-1, 1, (batch_size, noise_dim))
    y = np.ones(batch_size).astype(int)
    gan_loss = gan.train_on_batch(noise, y)

    # Train the discriminator on a mix of fake and real data
    data_batch = data[np.random.randint(num_examples, size=batch_size)]
    data_labels = np.ones(batch_size).astype(int)
    fake_data_batch = generator.predict(noise)
    fake_data_labels = np.zeros(batch_size).astype(int)

    X = np.concatenate((data_batch, fake_data_batch), axis=0)
    y = np.concatenate((data_labels, fake_data_labels))

    discriminator_loss = discriminator.train_on_batch(X, y)

    if step % 100 == 0:
        print step, gan_loss, discriminator_loss
