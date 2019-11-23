import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import tensorflow.compat.v1 as tf
import pickle as pkl
from tqdm import trange
import time
from imageio import imread
from skimage.transform import resize
from skimage import io
tf.disable_eager_execution()

# #data = np.load("airplane.npy")
#
#
# # for file in datasets:
# #     file = "/Users/darklord/Downloads/datasets/" + file
# #     data = np.load(file)
# #     fig = plt.figure(figsize=(10, 10))
# #     columns = 15
# #     rows = 10
# #     for i in range(1, columns * rows):
# #         img = data[i + 44].reshape((28, 28))
# #         fig.add_subplot(rows, columns, i)
# #         plt.imshow(img)
# #     plt.show()

def model_inputs(real_dims, z_dims):
    inputs_real = tf.placeholder(tf.float32, shape=(None, real_dims), name='input_real')
    inputs_z = tf.placeholder(tf.float32, shape=(None, z_dims), name='input_z')

    return inputs_real, inputs_z


def get_data(path):
    try:
        data = np.load(path)

        Y = []
        for i in trange(data.shape[0]):
            Y.append([1, 0])
        Y = np.array(Y)

        (x_train, y_train, x_test, y_test) = train_test_split(data, Y)
        x_train = (x_train.astype(np.float32)) / 255
        x_train = x_train.reshape(x_train.shape[0], 784)

        return (x_train, y_train, x_test, y_test)

    except Exception as e:
        print(e)


x_train, y_train, x_test, y_test = get_data("/home/rharode/Documents/dataset/wine_bottle.npy")


def generator(z, out_dims, n_units=128, reuse=False, alpha=0.01):
    with tf.variable_scope('generator', reuse=reuse):
        # hidden layer
        h1 = tf.layers.dense(z, n_units, activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1, n_units, activation=tf.nn.leaky_relu)

        # leaky relu implementation
        # h1 = tf.maximum(alpha * h1, h1)

        # tanh
        logits = tf.layers.dense(h2, out_dims)

        out = tf.tanh(logits)

        return out


def discriminator(x, n_units=128, reuse=False, alpha=0.01):
    with tf.variable_scope('discriminator', reuse=reuse):
        # hidden layer
        h1 = tf.layers.dense(x, n_units, activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1, n_units, activation=tf.nn.leaky_relu)

        # sigmoid
        logits = tf.layers.dense(h2, 1, activation=None)
        out = tf.sigmoid(logits)

        return out, logits

def save_gen_images(epoch,samples,save=False):
  fig, axes = plt.subplots(figsize=(5, 5), nrows=1, ncols=1)
  for img in samples[epoch-1]:
      plt.imshow(img.reshape((28, 28)), cmap='Greys')
      plt.axis('off')
      if save and epoch is not None:
          plt.savefig('/home/rharode/Documents/gan_winebottle/image_at_epoch_{:03d}.png'.format(epoch))


def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(5, 5), nrows=4, ncols=4)
    for ax,img in zip(axes.flatten(), samples[epoch]):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(img.reshape((28, 28)), cmap='Greys')

    plt.savefig('/home/rharode/Documents/collection_gan_winebottle/samples_at_epoch_{:3d}.png'.format(epoch))

r_size = 784
z_size = 100
g_units = 128
d_units = 128
lr = 0.002

tf.reset_default_graph()

inputs_real, inputs_z = model_inputs(r_size, z_size)

g_out = generator(inputs_z, r_size, g_units)
d_out_real, real_logit = discriminator(inputs_real, )
d_out_fake, fake_logits = discriminator(g_out, reuse=True)

d_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logit, labels=tf.ones_like(real_logit) * (1 - 0.1)))
d_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits)))
d_loss = d_loss_fake + d_loss_real
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like(fake_logits)))

tvar = tf.trainable_variables()
gvar = [var for var in tvar if 'gen' in var.name]
dvar = [var for var in tvar if 'dis' in var.name]

#Optimizer
d_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(d_loss, var_list=dvar)
g_opt = tf.train.AdamOptimizer(lr).minimize(g_loss, var_list=gvar)

start = time.time()
batch_size = 100
epochs = 501
samples = []
losses_g = []
losses_d = []

init = tf.global_variables_initializer()

# Only save generator variables
saver = tf.train.Saver(var_list=gvar)

with tf.Session() as sess:
    sess.run(init)
    for e in range(1,epochs):
        for i in range(x_train.shape[0] // batch_size):
            batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

            batch_images = 2*batch - 1

            # Sample random noise for G
            batch_z = np.random.uniform(-1, 1, size=(batch_size, z_size))

            # Run optimizers
            _ = sess.run(d_opt, feed_dict={inputs_real: batch_images, inputs_z: batch_z})
            _ = sess.run(g_opt, feed_dict={inputs_z: batch_z})

        # At the end of each epoch, get the losses and print them out
        train_loss_d = sess.run(d_loss, {inputs_z: batch_z, inputs_real: batch_images})
        train_loss_g = g_loss.eval({inputs_z: batch_z})
        if e % 10 == 0:
            print("Epoch {}/{}...".format(e, epochs),
                  "Discriminator Loss: {:.4f}...".format(train_loss_d),
                  "Generator Loss: {:.4f}".format(train_loss_g))

        # Save losses
        losses_g.append(train_loss_g)
        losses_d.append(train_loss_d)


        sample_z = np.random.uniform(-1, 1, size=(16, z_size))
        gen_samples = sess.run(
            generator(inputs_z, r_size, n_units=g_units, reuse=True, alpha=0.01),
            feed_dict={inputs_z: sample_z})
        samples.append(gen_samples)
        if e % 50 == 0 or e > 450:
            save_gen_images(e, samples, save=True)

    plt.plot(range(epochs), losses_g)
    plt.plot(range(epochs), losses_d)
    plt.legend(['Generator Loss', 'Discriminator Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator vs Discriminator for Wine Bottle category')

# Save training generator samples
with open('gen_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)


with open('gen_samples.pkl', 'rb') as f:
    samples = pkl.load(f)

epoch_list = [20,50,100,300,500]
for epoch in epoch_list:
    _ = view_samples(epoch, samples)

# x = imread('./genimages/Unknown-5.png',pilmode = 'L')
# x = resize(x,((28,28)))
# io.imshow(x)
# io.show()
# print(x.shape)
# x = np.reshape(x,(28,28,1))
# print(x.shape)


#
# def view_samples(epoch, samples):
#     fig, axes = plt.subplots(figsize=(5, 5), nrows=4, ncols=4, sharey=True, sharex=True)
#     i= 0
#     for img in samples[epoch]:
#         i = i+1
#         plt.imshow(img.reshape((28,28)),cmap = 'Greys')
#         plt.savefig('./collection_genimages/sample{:2d}_at_epoch_{:03d}.png'.format(i,epoch))

# rows, cols = 10, 10
# fig, axes = plt.subplots(figsize=(8, 6), nrows=rows, ncols=cols, sharex=True, sharey=True)
#
# for sample, ax_row in zip(samples[::int(len(samples) / rows)], axes):
#     for img, ax in zip(sample[::int(len(sample) / cols)], ax_row):
#         ax.imshow(img.reshape((28, 28)), cmap='Greys')
#         ax.xaxis.set_visible(False)
#         ax.yaxis.set_visible(False)