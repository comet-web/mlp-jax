# using jax to hallucinate the original image > restore a version as perfect as the original version


import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

def img_create(size = 100):
    x = jnp.linspace(-1, 1, size)
    y = jnp.linspace(-1, 1, size)
    X, Y  = jnp.meshgrid(x, y) # this shows the location of each and every pixel on the image

    # here we make a simple circle > we paint it using code 
    radius = jnp.sqrt(X**2 + Y**2) # this calculate the distance of every point from the origin (0,0) 
    imgv1 = jnp.where(radius < 0.5, 1.0, 0.0) # this says if (radius < 0.5 ) then paint the pixel with white(1) else with black (0)
    return imgv1

size = 100
imgv1 = img_create(size)

key = jax.random.PRNGKey(0)
noise = jax.random.normal(key, (size, size)) * 0.3
# here we initialise a matrix with the same size of original image with random numbers and punish it to give some valid numbers that lie in an acceptable range.

imgv2 = imgv1 + noise

# now we find the noise of the corrupted / noise added image. 
def totVarLoss(img):
    # we need to hypothesize that for the clean image, every pixel is similar to its neighbors
    # whereas for the noisy image it is very varying.
    diff_x = img[:, 1:] - img[:, :-1]
    diff_y = img[1:, :] - img[:-1, :]
    return jnp.sum(jnp.abs(diff_x)) + jnp.sum(jnp.abs(diff_y))

def loss_fn(canvas, target_image):
    similarity_loss = jnp.sum((canvas - target_image) ** 2)
    smoothness_loss = totVarLoss(canvas)
    return similarity_loss + 2.0 * smoothness_loss

# here 2 is the hyperparameter that we can tune, this says smoothness is twice as important as matching the original pixels

@jax.jit
def update(canvas, target, lr = 0.01):
    grads = jax.grad(loss_fn)(canvas, target)
    new_canvas = canvas - lr*grads
    return new_canvas

current_canvas = jnp.array(imgv2)

for i in range(1001):
    current_canvas = update(current_canvas, imgv2)   # this talks about the 1000 iterations.

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(imgv1, cmap='gray')
ax[0].set_title("ORIGINAL IMAGE")
ax[0].axis('off')

ax[1].imshow(imgv2, cmap='gray')
ax[1].set_title("NOISY IMAGE")
ax[1].axis('off')

ax[2].imshow(current_canvas, cmap='gray')
ax[2].set_title("RESTORED IMAGE")
ax[2].axis('off')

plt.show()