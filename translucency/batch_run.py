




if __name__ == '__main__':

    #latent_dims = [64, 128]

    #from run_latent_decoder import *
    #batch_run_make_latent_decoder(latent_dims, cond='obj_mask', cond2='ae_')

    from run_transfer_generation import *
    latent_dims = [2, 4, 16, 64, 128, 256]
    batch_run_make_swapping(latent_dims, cond='obj_mask', cond2='')