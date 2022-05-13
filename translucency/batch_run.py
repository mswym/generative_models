




if __name__ == '__main__':
    #latent_dims = [16, 256, 2]

    #from ae import *
    #batch_run_ae(latent_dims)

    #from gan_dcgan import *
    #batch_run_dcgan(latent_dims)

    latent_dims = [128]
    from ae import *
    batch_run_ae(latent_dims)


    latent_dims = [2, 4, 8, 16, 32, 64, 128, 256]

    from run_latent_decoder import *
    batch_run_make_latent_decoder(latent_dims)


    #from gan_dcgan import *
    #batch_run_dcgan(latent_dims)