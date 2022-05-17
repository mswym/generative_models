




if __name__ == '__main__':

    latent_dims = [16, 256, 2, 4, 8, 32, 64, 128]

    from run_latent_decoder import *
    batch_run_make_latent_decoder(latent_dims, cond='obj_mask', cond2='')

    from run_transfer_generation import *
    batch_run_make_swapping(latent_dims, cond='obj_mask', cond2='')