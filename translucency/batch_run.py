


if __name__ == '__main__':
    latent_dims = [16, 256, 2]

    from ae import *
    for latent_dim in latent_dims:
        batch_run_ae(latent_dim)

    from gan_dcgan import *
    for latent_dim in latent_dims:
        batch_run_dcgan(latent_dims)

    latent_dims = [4, 8, 32, 64, 128]

    from ae import *
    for latent_dim in latent_dims:
        batch_run_ae(latent_dims)

    from gan_dcgan import *
    for latent_dim in latent_dims:
        batch_run_dcgan(latent_dims)