from ptyrannosaur.models.autoencoder import Autoencoder
import json
import shutil
import os
import pickle
import ptyrannosaur.engine.utils as utils
import ptyrannosaur.engine.dataset as dataset
import ptyrannosaur.engine.stitching as stitching
import matplotlib.pyplot as plt
import numpy as np
import argparse
import h5py
import time
import tifffile
from matplotlib.patches import Circle

def load_model(load_path):
    """Load model and optimizer state."""
    with open(load_path, 'rb') as file:
        model_data = pickle.load(file)
        model_state = model_data['model_state']
        sim_config = model_data['simulation_config']
        train_config = model_data['training_config']
    return model_state, sim_config, train_config

def main(model_path, eval_path):
    """Evaluate a network based on configuration in json at a target path."""
    print('Starting evaluation')
    with open(os.path.join(eval_path,'experimental_config.json'), 'r') as f:
        eval_config = json.load(f)

    eval_settings = eval_config['settings']
    eval_params = eval_config['evaluation_parameters']

    model_state, sim_config, train_config = load_model(model_path)
    s_params = sim_config['simulation_parameters']
    t_params = train_config['training_parameters']

    # Save python files and json to save backups
    os.makedirs(os.path.join(eval_path,'metadata'), exist_ok=True)
    shutil.copy(os.path.join(eval_path,'experimental_config.json'), 
                os.path.join(eval_path,'metadata','experimental_config.json'))
    for filename in os.listdir(os.getcwd()):
        if filename.endswith(".py") or filename.endswith(".sbatch"):
            source_path = os.path.join(os.getcwd(), filename)
            destination_path = os.path.join(eval_path,'metadata', filename)
            shutil.copy(source_path, destination_path)
            print(f"Copied: {filename}",flush=True)
    
    # Define the model
    model = Autoencoder(t_params['num_down_blocks'],t_params['num_up_blocks'],
                        t_params['num_base_filters'],t_params['kernel_size'],
                        t_params['pooling_size'],t_params['upsample_size'],
                        t_params['mom'],t_params['leaky_val'],t_params['stride'],
                        t_params['out_layers'],t_params['out_size'])

    # Iterate through all folders
    subdirs = sorted([item for item in os.listdir(eval_settings['data_path']) if os.path.isdir(os.path.join(eval_settings['data_path'], item))])
    for subdir in subdirs:
        print(f'\nStarting to process {subdir}',flush=True)
        save_dir = os.path.join(eval_path,eval_settings['save_path'],subdir)
        os.makedirs(save_dir, exist_ok=True)
        print(f'Saving outputs to {save_dir}')
        start_time = time.perf_counter()
        cbeds, neighbors, scan_points = utils.process_exp_path(os.path.join(eval_settings['data_path'], subdir),eval_params['num_scans_1d'],t_params['n_k'],s_params['step_size'],s_params['output_d_x'],t_params['num_neighbors_1d'])
        load_time = time.perf_counter()
        print(f'\nData loaded in {load_time-start_time:.3f} secs',flush=True)

        # Run PtyRANNOSAUR
        print('\n\nBeginning to run PtyRANNOSAUR')
        batch_size = eval_params['num_scans_1d']-t_params['num_neighbors_1d']+1
        output_objs, scan_pts = dataset.loop_batch_exp(cbeds, neighbors, scan_points, batch_size, model, model_state)
        network_time = time.perf_counter()
        print(f'PtyRANNOSAUR run in {network_time-load_time:.3f} secs',flush=True)

        # Save patches
        if eval_params['save_patches']:
            with h5py.File(os.path.join(save_dir,'patches.h5'), "w") as h5f:
                h5f.create_dataset("Data", data=output_objs)
            print(f'\nOutput patches saved to {save_dir} as patches.h5')

        # Process the patches
        print('\n\nBeginning to perform rigid stitching')
        output_objs = np.array(output_objs)
        scan_pts = np.array(scan_pts)
        start_stitching_time = time.perf_counter()
        output_full_obj = stitching.grid_stitch(output_objs, scan_pts)[np.newaxis,:,:,:]
        eng_stitching_time = time.perf_counter()
        print(f'Rigid stitching complete in {eng_stitching_time - start_stitching_time:.3f} secs',flush=True)
        tifffile.imwrite(os.path.join(save_dir,'grid_stitched.tiff'), output_full_obj[0,:,:,0].astype(np.float32))
        print(f'Rigid stitching saved to {save_dir} as grid_stitched.tiff')

        # Position correct the patches
        if eval_params['learn_stitching']:
            print('\n\nBeginning to perform position correction')
            start_stitching_time = time.perf_counter()
            stitched_image, support, patch_pos = stitching.learn_stitch(output_objs, scan_pts)
            eng_stitching_time = time.perf_counter()
            print(f'Position corrected stitching complete in {eng_stitching_time - start_stitching_time:0.3f} secs',flush=True)
            tifffile.imwrite(os.path.join(save_dir,'stitched_output.tiff'), stitched_image.astype(np.float32))
            print(f'Position corrected stitching saved to {save_dir} stitched_output.tiff')

        # Create main figure
        main_fig, main_axs = plt.subplots(2, 3, figsize=(12, 9))
        sup_title_font_size = 24
        sub_title_font_size = 20
        # Reload cbed data
        data_file_path = os.path.join(eval_settings['data_path'], subdir)
        n_scans = eval_params['num_scans_1d']
        n_k = t_params['n_k']
        cbed_data_original = np.fromfile(data_file_path+f"/scan_x{n_scans}_y{n_scans}.raw", '<f4')
        cbed_data = cbed_data_original.reshape((n_scans, n_scans, n_k+2, n_k))
        cbed_data = cbed_data[:, :, :n_k, :]
        # Adjust data to match simulated data structure.
        cbed_data = np.flip(cbed_data,-2)
        center = ((n_k) / 2, (n_k-2) / 2)
        y, x = np.indices((n_k, n_k))
        r = np.sqrt((y - center[0])**2 + (x - center[1])**2)
        annulus = (r >= eval_params['virtual_image_min_radius']) & (r <= eval_params['virtual_image_max_radius'])
        # Integrate over detector pixels
        virtual_image = cbed_data[..., annulus].sum(axis=-1)

        # Virtual image
        main_axs[0, 0].imshow(virtual_image, cmap="gray")
        main_axs[0, 0].set_title("Virtual BF",fontsize=sub_title_font_size)
        main_axs[0, 0].axis("off")

        # Rigid stitching
        main_axs[0, 1].imshow(output_full_obj[0,:,:,0], cmap="gray")
        main_axs[0, 1].set_title("PtyRANNOSAUR\nRigid Stitching",fontsize=sub_title_font_size)
        main_axs[0, 1].axis("off")

        # Position corrected (optional)
        if eval_params["learn_stitching"]:
            main_axs[0, 2].imshow(stitched_image, cmap="gray")
            main_axs[0, 2].set_title("PtyRANNOSAUR\nPosition Corrected",fontsize=sub_title_font_size)
        else:
            main_axs[0, 2].axis("off")

        # Example CBED
        main_axs[1, 0].imshow(cbed_data[0,0], cmap="gray")
        main_axs[1, 0].set_title("Example CBED",fontsize=sub_title_font_size)
        main_axs[1, 0].axis("off")
        from matplotlib.patches import Circle
        
        # Create the circle patches (center is already defined as (x, y) above)
        outer_circle = Circle(center, eval_params['virtual_image_max_radius'], edgecolor='red', facecolor='none', linestyle='--', linewidth=1.5, label='Virtual BF Area')
        main_axs[1, 0].add_patch(outer_circle)
        if eval_params['virtual_image_min_radius'] != 0:
            inner_circle = Circle(center, eval_params['virtual_image_min_radius'], edgecolor='cyan', facecolor='none', linestyle='--', linewidth=1.5, label='Virutal Image Min Radius')
            main_axs[1, 0].add_patch(inner_circle)
        main_axs[1, 0].legend()

        # PACBED
        main_axs[1, 1].imshow(np.log(cbed_data.mean(axis=(0,1))), cmap="gray")
        main_axs[1, 1].set_title("log(PACBED)",fontsize=sub_title_font_size)
        main_axs[1, 1].axis("off")


        # Corrected positions (optional)
        if eval_params["learn_stitching"]:
            main_axs[1, 2].imshow(support, cmap="gray")
            data = output_objs.reshape(batch_size, batch_size, 30, 30)
            offset = (np.array(data.shape[2:]) - 1) / 2
            pos_plot = patch_pos + offset[:, np.newaxis]
            main_axs[1, 2].plot(pos_plot[1],pos_plot[0],".",color="red",markersize=1,alpha=0.6)
            main_axs[1, 2].set_title("PtyRANNOSAUR\nCorrected Positions",fontsize=sub_title_font_size)
        else:
            main_axs[1, 2].axis("off")

        # Remove axes from any image that is shown
        for ax in main_axs.flat:
            if ax.axison:
                ax.set_xticks([])
                ax.set_yticks([])

        main_fig.suptitle("PtyRANNOSAUR outputs", fontsize=sup_title_font_size, y=1.0)
        plt.savefig(os.path.join(save_dir,'PtyRANNOSAUR_output.png'), bbox_inches='tight',dpi=300)
        print(f'\n\nOutput figure saved to {save_dir} as PtyRANNOSAUR_output.png')
        plt.close()


        # Save examples patches
        n_patches = 3
        fig, axes = plt.subplots(n_patches,n_patches, figsize=(9, 9))
        batch_size
        for i in range(n_patches):
            for j in range(n_patches):
                _ = axes[j,i].imshow(output_objs[i + batch_size*j,:,:,0],cmap='gray')
                axes[j,i].set_xticks([])
                axes[j,i].set_yticks([])
        for ax in axes.flat:
            ax.label_outer()
        fig.suptitle("Example patches", fontsize=sup_title_font_size, y=0.98)
        plt.savefig(os.path.join(save_dir,'patch_examples.png'), bbox_inches='tight',dpi=300)
        plt.close()

        # Save example cbeds
        fig, axes = plt.subplots(5, 5, figsize=(9,9.5))
        reshaped_dps = cbeds[neighbors[0:1]].reshape(5,5,128,128)

        for i in range(5):
            for j in range(5):
                _ = axes[i,j].imshow(reshaped_dps[i,j,:,:],cmap='gray')
                axes[i,j].set_xticks([])
                axes[i,j].set_yticks([])

        fig.suptitle("Example CBEDS", fontsize=sup_title_font_size, y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,'cbed_examples.png'), bbox_inches='tight',dpi=300)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run simulation for set of materials.')
    parser.add_argument('--model_path', type=str, required=True, help='model path')
    parser.add_argument('--eval_path', type=str, required=True, help='eval path')
    args = parser.parse_args()
    main(model_path=args.model_path, eval_path=args.eval_path)

    