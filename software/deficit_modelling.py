import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

deficits = ['hearing', 'language', 'introspection', 'cognition', 'mood', 'memory', 'aversion', 'coordination', 'interoception', 'sleep', 'reward', 'visual recognition', 'visual perception', 'spatial reasoning', 'motor', 'somatosensory']

class deficit_inference:
    def __init__(self, roipath, input_type, roi_thresh, images_loaded, disco_thresh=0.5):
        self.roipath = roipath
        self.input_type = input_type
        self.roi_thresh = roi_thresh
        self.images_loaded = images_loaded
        self.disco_thresh = disco_thresh

        self.roipairs = {}
        for roipair in range(1, 17):
            rois = os.listdir(self.roipath)
            roipair_str = str(roipair) + '_'
            file = False
            for roi in rois:
                if roi.startswith(roipair_str):
                    if 'nii' in roi:
                        file = roi
            if file:
                roi_pair = nib.load(os.path.join(roipath, file)).get_fdata()
            else:
                raise ValueError("File not found")

            self.roipairs[roipair] = roi_pair

        self.deficits = deficits


    def find_deficits(self, df):
        n_deficits = len(os.listdir(self.roipath))
        preallocate_regionmasks = {}
        for roipair in range(1, n_deficits + 1):
            roi_pair = self.roipairs[roipair]

            # Create binary masks for each ROI within the pair.
            roi1 = (roi_pair == 1).astype(int)
            roi2 = (roi_pair == 2).astype(int)
            regionmasks = [roi1, roi2]

            # Now compute overlap between each lesion and the ROI mask. Exceeding a
            # threshold by proportion of the ROI will register is a 'hit' and the
            # associated deficit can therefore be simulated.
            roi_vols = [np.sum(regionmasks[region]) for region in range(len(regionmasks))]
            treatments = ["1", "0"] if np.argmax(roi_vols) == 0 else ["0", "1"]

            inds1 = np.where(roi1 == 1)
            centroid1_x = inds1[0].mean()
            centroid1_y = inds1[1].mean()
            centroid1_z = inds1[2].mean()

            inds2 = np.where(roi2 == 1)
            centroid2_x = inds2[0].mean()
            centroid2_y = inds2[1].mean()
            centroid2_z = inds2[2].mean()

            roi_centroids_x = [centroid1_x, centroid2_x]
            roi_centroids_y = [centroid1_y, centroid2_y]
            roi_centroids_z = [centroid1_z, centroid2_z]

            preallocate_regionmasks[roipair] = [regionmasks, roi_vols, treatments, (roi_centroids_x, roi_centroids_y, roi_centroids_z)]

        df.reset_index(inplace=True, drop=True)
        for i in trange(len(df)):
            img = self.images_loaded[df['filename'].iloc[i]]
            for roipair in range(1, n_deficits + 1):
                regionmasks, roi_vols, treatments, centroids = preallocate_regionmasks[roipair]
                for region, vol, treatment in zip(regionmasks, roi_vols, treatments):
                    treatment_susceptibility = f"{self.deficits[roipair - 1]}_W{treatment}"
                    thresh = int(np.round(vol * self.roi_thresh))
                    if np.sum(((img > self.disco_thresh).astype(int) * region)) > thresh:
                        df.loc[i, treatment_susceptibility] = 1
                    else:
                        df.loc[i, treatment_susceptibility] = 0

        return df, preallocate_regionmasks



def harmonize_columns(main_df, other_dfs_path, latent_list, train_or_test, kf_count, input_type, reductions):

    main_df.reset_index(inplace=True, drop=True)

    other_latent_dfs = {dim: get_file(other_dfs_path, f"{train_or_test}_{kf_count}_['{input_type}']_dim_{dim}") for dim in latent_list[1:]}
    other_reductions = {REDUCTION: pd.DataFrame() for REDUCTION in reductions}
    for i in trange(len(main_df)):
        img_name = main_df["filename"].iloc[i]
        for dim in latent_list[1:]:
            gt_train_dim = other_latent_dfs[dim]
            include = gt_train_dim.loc[gt_train_dim['filename'] == img_name]
            for REDUCTION in reductions:
                if len(include):
                    other_reductions[REDUCTION] = pd.concat([other_reductions[REDUCTION],
                                                             pd.DataFrame({f"{REDUCTION}_{input_type}_{dim}_K{kf_count}": [include[f"{REDUCTION}_{input_type}_{dim}_K{kf_count}"].item()]}, index=[i])])
                else:
                    print('unknown')
                    other_reductions[REDUCTION] = pd.concat([other_reductions[REDUCTION],
                                                             pd.DataFrame({f"{REDUCTION}_{input_type}_{dim}_{kf_count}": np.nan}, index=[i])])

    for REDUCTION in reductions:
        for col in other_reductions[REDUCTION].columns:
            main_df[col] = other_reductions[REDUCTION].loc[other_reductions[REDUCTION][col].isna() == False][col]

    return main_df


def get_file(path, name):
    loadfiles = os.listdir(path)
    filetoload = 0
    for file in loadfiles:
        if file.startswith(name):
            filetoload = file
    return pd.read_pickle(os.path.join(path, filetoload)) if filetoload else None

def run(parameters):
    paths, ground_truth, reductions = parameters
    (lesionpath, discopath, path) = paths
    (latent_list, kfold_deficits, roi_threshs, names) = ground_truth
    (run_ae, run_vae, run_nmf, run_pca) = reductions

    input_types = []
    if lesionpath: input_types.append('lesion')
    if discopath: input_types.append('disco')

    reductions_list = []
    if run_vae: reductions_list.append('vae_means')
    if run_ae: reductions_list.append('ae')
    if run_nmf: reductions_list.append('nmf')
    if run_pca: reductions_list.append('pca')

    roi_stem = os.path.join(os.getcwd(), "../../atlases", "2mm_parcellations")
    roipaths = [os.path.join(roi_stem, name) for name in names]
    for roi_thresh in roi_threshs:
        for input_type in input_types:
            image_path = lesionpath if input_type == 'lesion' else discopath
            images_list = os.listdir(image_path)
            images_loaded = {filename: nib.load(os.path.join(image_path, filename)).get_fdata() for filename in tqdm(images_list)}
            for kf_count in range(kfold_deficits):
                for roipath, name in zip(roipaths, names):
                    generator = deficit_inference(roipath, input_type, roi_thresh, images_loaded)
                    savepath = os.path.join(path, f"{input_type}_{roi_thresh}_{name}_{kf_count}")
                    if not os.path.exists(savepath): os.makedirs(savepath)

                    ground_truth_train = get_file(path, f"train_{kf_count}_dim_{latent_list[0]}_['{input_type}']")
                    ground_truth_train, susceptibility_information_train = generator.find_deficits(ground_truth_train)
                    ground_truth_train = harmonize_columns(ground_truth_train, path, latent_list, "train", kf_count, input_type, reductions_list)
                    ground_truth_train.to_pickle(os.path.join(savepath, f"train_{input_type}_{roi_thresh}_{kf_count}.pkl"))
                    del ground_truth_train
                    centroids = {roi: susceptibility_information_train[roi][-1] for roi in range(1, len(deficits) + 1)}
                    pd.DataFrame(centroids).to_json(os.path.join(savepath, "centroids.json"))

                    ground_truth_test = get_file(path, f"test_{kf_count}_dim_{latent_list[0]}_['{input_type}']")
                    ground_truth_test, _ = generator.find_deficits(ground_truth_test)
                    ground_truth_test = harmonize_columns(ground_truth_test, path, latent_list, "test", kf_count, input_type, reductions_list)
                    ground_truth_test.to_pickle(os.path.join(savepath, f"test_{input_type}_{roi_thresh}_{kf_count}.pkl"))
                    del ground_truth_test

def command_line_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="", help="Path")
    parser.add_argument("--lesionpath", type=str, default="", help="Path to lesion nii files")
    parser.add_argument("--discopath", type=str, default="", help="Path to disconnectome nii files")

    parser.add_argument("--latent_list", type=int, nargs='+', default=[2, 4, 8, 16, 32, 64, 128, 256], help="Number of folds for cross-validation")
    parser.add_argument("--kfold_deficits", type=int, default=10, help="K-fold crossval")
    parser.add_argument("--roi_threshs", type=int, nargs='+', default=[0.05], help="Threshold for susceptibility modelling")
    parser.add_argument("--names", type=str, nargs='+', default=["genetics", "receptor"], help="Ground truth rationales")

    parser.add_argument("--run_ae", type=bool, default=False, help="Run autoencoder")
    parser.add_argument("--run_vae", type=bool, default=False, help="Run variational autoencoder")
    parser.add_argument("--run_nmf", type=bool, default=True, help="Run non-negative matrix factorisation")
    parser.add_argument("--run_pca", type=bool, default=True, help="Run principal component analysis")

    args = parser.parse_args()
    paths = (args.lesionpath, args.discopath, args.path)
    ground_truth = (args.latent_list, args.kfold_deficits, args.roi_threshs, args.names)
    reductions = (args.run_ae, args.run_vae, args.run_nmf, args.run_pca)
    return (paths, ground_truth, reductions)

if __name__ == "__main__":
    parameters = command_line_options()
    run(parameters)