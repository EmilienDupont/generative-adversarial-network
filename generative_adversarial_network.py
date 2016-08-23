import matplotlib as plt
import numpy as np
import plotly.graph_objs as go
import plotly.offline as py

from keras.layers import Dense, Input
from keras.layers import Convolution1D
from keras.models import Model
from keras.optimizers import Adam
from plotly import tools

## Model parameters
data_dim = 64
noise_dim = 64

## Training parameters
batch_size = 32
num_steps = 10000
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
x = Dense(128, activation='relu')(discriminator_input)
x = Dense(128, activation='relu')(x)
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
noise = np.random.uniform(-1, 1, (num_examples, noise_dim))

# Pre train discriminator
X = np.concatenate((data, generator.predict(noise)), axis=0)
data_labels, fake_data_labels = np.ones(num_examples).astype(int), np.zeros(num_examples).astype(int)
y = np.concatenate((data_labels, fake_data_labels))
discriminator_loss = discriminator.fit(X, y, nb_epoch=2)

## Training
for step in range(num_steps):
    # Train the discriminator on a mix of fake and real data
    for _ in range(2):
        noise_batch = noise[np.random.randint(num_examples, size=batch_size)]
        data_batch = data[np.random.randint(num_examples, size=batch_size)]
        data_labels = np.ones(batch_size).astype(int)
        fake_data_batch = generator.predict(noise_batch)
        fake_data_labels = np.zeros(batch_size).astype(int)
        X = np.concatenate((data_batch, fake_data_batch), axis=0)
        y = np.concatenate((data_labels, fake_data_labels))
        discriminator_loss = discriminator.train_on_batch(X, y)

    # Train the generator
    y = np.ones(batch_size).astype(int)
    gan_loss = gan.train_on_batch(noise_batch, y)

    if step % 500 == 0:
        print "step", step
        print "\t", "gan_loss:", gan_loss, "discriminator_loss", discriminator_loss

## Plotting
noise_batch = noise[np.random.randint(num_examples, size=256)]
fake_data_batch = generator.predict(noise_batch)
data_batch = data[np.random.randint(num_examples, size=256)]

# Plot distributions
trace_fake = go.Scatter(
    x = fake_data_batch[:,0],
    y = fake_data_batch[:,1],
    mode = 'markers'
)

trace_real = go.Scatter(
    x = data_batch[:,0],
    y = data_batch[:,1],
    mode = 'markers'
)

data = [trace_fake, trace_real]
fig = go.Figure(data=data)
py.plot(fig)
