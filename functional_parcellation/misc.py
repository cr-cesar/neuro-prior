import argparse
import copy
import os
import statistics
import xml.etree.cElementTree as et

import colorcet as cc
import joblib
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.cm
import matplotlib.colors as mcolors
from neuroquery import datasets
from neuroquery.encoding import NeuroQueryModel
import nibabel as nib
from nilearn import image, plotting
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from scipy import ndimage as ndi
from scipy.ndimage.morphology import binary_dilation
from skimage.measure import label
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity


def cmdargs_func_gene():
    """
    Imports variables parsed from the command line arguments interface.

    Returns
    -------

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist_thresh', '--dist_thresh', type=int, default=0,
                        help='Specify the clustering distance threshold. Otherwise it will be selected to accomodate '
                             'the number of groups desired.')
    parser.add_argument('--grey_matter_thresh', '--grey_matter_thresh', type=float, default=0.2,
                        help='Threshold of tissue probability map for grey matter to include voxel in parcellation.')
    parser.add_argument('--white_matter_thresh', '--white_matter_thresh', type=float, default=0.2,
                        help='White matter mask saved for paraview visualisation.')
    parser.add_argument('--figsize', '--figsize', type=tuple, default=(8, 6),
                        help='Default figure size.')
    parser.add_argument('--cmap', '--cmap', type=str, default='turbo',
                        help='MatPlotLib or colorcet compatible colourmap. This will apply to the exported figures '
                             'and colourmaps for the parcellation, dendrogram, term embeddings, term lists.')
    parser.add_argument('--notes', '--notes', type=str, default='None',
                        help='Any notes to be in parameters doc in file.')
    parser.add_argument('--site', '--site', type=str, default='hpc',
                        help='Path initialisation from local or high-performance cluster?')
    parser.add_argument('--dimred', '--dimred', type=int, default=0,
                        help='Functional data dimensionality reduction. Parameter options are 0: None, 1: PCA, 2: NMF')
    parser.add_argument('--dimred_components', '--dimred_components', type=int, default=50,
                        help='Number of components to reduce each voxel to by method assigned by --dimred.')
    parser.add_argument('--z_thresh', '--z_thresh', type=float, default=0,
                        help='Threshold to cut-off Z values in functional data before dimensionality reduction. If 0, '
                             'the raw data will be used')
    parser.add_argument('--n_terms', '--n_terms', type=int, default=10,
                        help='Number of neurological terms by similarity to use in identification of functional theme '
                             'when topic modelling is used.')
    parser.add_argument('--topn', '--topn', type=int, default=10,
                        help='Number of similar terms to be extracted using topic modelling '
                             'in identification of functional theme.')
    parser.add_argument('--loadpath', '--loadpath', type=str, default="False",
                        help='Path of pretrained clustering model, to load.')
    parser.add_argument('--n_cluster_min', '--n_cluster_min', type=int, default=2,
                        help='Process and save results including and higher than this number of functional groups.')
    parser.add_argument('--n_cluster_max', '--n_cluster_max', type=int, default=24,
                        help='Process and save results up to this number of clusters.')
    parser.add_argument('--func_split', '--func_split', type=int, default=0,
                        help='Split ROIs by downstream hierarchical clustering of functional data or if number of '
                             'connected components is 2. 1 to do so, 0 to skip to genetics only.')
    parser.add_argument('--similarity_measure', '--similarity_measure', type=str, default='dot',
                        help='Dot product or cosine similarity, as the similarity measure to match terms to '
                             'functional map loci.')
    parser.add_argument('--knn', '--knn', type=int, default=5,
                        help='Number of nearest neighbours to average to extrapolate in dilated mask for Surf-Ice '
                             'visualisation, and for reclassification of small groups.')
    parser.add_argument('--target_mni_dim', '--target_mni_dim', type=tuple, default=(91, 109, 91),
                        help='Export outputs will also be resampled to this shape if different from input.')
    parser.add_argument('--term_exclusivity', '--term_exclusivity', type=int, default=2,
                        help='Index functional terms by similarity one group only? 0 for no, 1 for yes, '
                             'anything else for both.')

    # Read all into memory
    args = parser.parse_args()
    dist_thresh = args.dist_thresh
    grey_matter_thresh = args.grey_matter_thresh
    white_matter_thresh = args.white_matter_thresh
    figsize = args.figsize
    colourmap = args.cmap
    site = args.site
    notes = args.notes
    dimred = args.dimred
    dimred_components = args.dimred_components
    z_thresh = args.z_thresh
    n_terms = args.n_terms
    topn = args.topn
    loadpath = args.loadpath
    n_cluster_min = args.n_cluster_min
    n_cluster_max = args.n_cluster_max
    knn = args.knn
    similarity_measure = args.similarity_measure
    func_split = args.func_split
    target_mni_dim = args.target_mni_dim
    term_exclusivity = args.term_exclusivity

    if dimred == 0:
        dimred = False
    elif dimred == 1:
        dimred = 'PCA'
    elif dimred == 2:
        dimred = 'NMF'
    if z_thresh == 0:
        z_thresh = False
    loadpath = False if loadpath.lower() == 'false' else loadpath
    func_split = True if func_split == 1 else False
    if term_exclusivity == 1:
        term_exclusivity = True
    elif term_exclusivity == 0:
        term_exclusivity = False
    else:
        term_exclusivity = np.nan
    return z_thresh, dist_thresh, grey_matter_thresh, white_matter_thresh, dimred, dimred_components, n_cluster_min, n_cluster_max, knn, \
           similarity_measure, figsize, colourmap, site, n_terms, topn, func_split, loadpath, notes, target_mni_dim, term_exclusivity


def get_border_index(lst):
    """
    Returns list of indices for which the value varies from its previous state

    Parameters
    ----------
    a : array
        Array to export indices that vary from previous state

    Returns
    -------
    result : list
        List of indices that vary from the previous state

    Example
    -------
    get_border_index([2, 2, 2, 3, 4, 4, 5, 6])
    [3, 4, 6, 7]
    """
    last_value = None
    result = []
    for i, v in enumerate(lst):
        if v != last_value:
            last_value = v
            result.append(i)
    return result


def rescale_affine(input_affine, voxel_dims=[2, 2, 2], target_center_coords=None):
    """
    Uses a generic approach to rescaling an affine to arbitrary voxel dimensions.
    It allows for affines with off-diagonal elements by decomposing the affine
    matrix into u,s,v (or rather the numpy equivalents) and applying the scaling
    to the scaling matrix (s).

    Parameters
    ----------
    input_affine : np.array of shape 4,4
        Result of nibabel.nifti1.Nifti1Image.affine
    voxel_dims : list
        Length in mm for x, y, and z dimensions of each voxel.
    target_center_coords : list of float
        3 numbers to specify the translation part of the affine if not using the
        same as the input_affine.

    Returns
    -------
    target_affine : 4x4matrix
        The resampled image.
    """
    # Initialize target_affine
    target_affine = input_affine.copy()

    # Decompose the image affine to allow scaling
    u, s, v = np.linalg.svd(target_affine[:3, :3], full_matrices=False)

    # Rescale the image to the appropriate voxel dimensions
    s = voxel_dims

    # Reconstruct the affine
    target_affine[:3, :3] = u @ np.diag(s) @ v

    # Set the translation component of the affine computed from the input
    # image affine if coordinates are specified by the user.
    if target_center_coords is not None:
        target_affine[:3, 3] = target_center_coords

    return target_affine


def hex2rgb(hex):
    hex = hex.lstrip('#')
    return tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))


class func_parc:
    def __init__(self,
                 z_thresh,
                 dist_thresh,
                 grey_matter_thresh,
                 white_matter_thresh,
                 dimred,
                 dimred_components,
                 n_cluster_max,
                 knn,
                 similarity_measure,
                 figsize,
                 colourmap,
                 site,
                 n_terms,
                 topn,
                 func_split,
                 loadpath,
                 notes,
                 target_mni_dim,
                 term_exclusivity):
        self.z_thresh = z_thresh
        self.dist_thresh = dist_thresh
        self.grey_matter_thresh = grey_matter_thresh
        self.white_matter_thresh = white_matter_thresh
        self.dimred = dimred
        self.dimred_components = dimred_components
        self.n_cluster_max = n_cluster_max
        self.knn = knn
        self.similarity_measure = similarity_measure
        self.figsize = figsize
        self.colourmap = colourmap
        self.site = site
        self.n_terms = n_terms
        self.topn = topn
        self.func_split = func_split
        self.loadpath = loadpath
        self.notes = notes
        self.target_mni_dim = target_mni_dim
        self.term_exclusivity = term_exclusivity

    def init_paths(self, save_stem, func_path, tpm_path, gene_paths, emb_path):
        """

        Parameters
        ----------
        save_stem
        func_path
        tpm_path
        gene_paths
        emb_path

        Returns
        -------

        """
        self.save_stem = os.path.join(save_stem, f"grey_thresh_{self.grey_matter_thresh}_z_thresh_{self.z_thresh}")
        self.func_path = func_path
        self.tpm_path = tpm_path
        self.gene_paths = gene_paths
        self.emb = np.load(emb_path)

        self.savepath = os.path.join(self.save_stem,
                                     f'dist_{self.dist_thresh}_mask_{self.dimred}_{self.dimred_components}_n_terms_{self.n_terms}_topn_{self.topn}/')

        # If the same parameter dir already exists, a folder with the date and time appended
        # will be used as the savepath
        try:
            os.makedirs(self.savepath)
        except Exception as err:
            print(err)
            import datetime

            now = datetime.datetime.now()
            self.savepath = os.path.join(self.savepath, f'time_{now.strftime("%H")}_{now.strftime("%M")}_'
                                                        f'date_{now.strftime("%d")}_{now.strftime("%m")}_{now.strftime("%y")}/')
            os.makedirs(self.savepath)

        # Save a .txt file with parameters for importing to local post-processing
        # and to keep a record of the parameters with the results.
        params = {'z_thresh': self.z_thresh,
                  'dist_thresh': self.dist_thresh,
                  'grey_matter_thresh': self.grey_matter_thresh,
                  'white_matter_thresh': self.white_matter_thresh,
                  'dimred': self.dimred,
                  'dimred_components': self.dimred_components,
                  'n_cluster_max': self.n_cluster_max,
                  'figsize': self.figsize,
                  'local_or_hpc': self.site,
                  'n_terms': self.n_terms,
                  'topn': self.topn,
                  'loadpath': self.loadpath,
                  'notes': self.notes}
        param_file = open(os.path.join(self.savepath, "params.txt"), "w")
        param_file.write(repr(params))
        param_file.close()

        dir_list = os.listdir(self.func_path)
        function_map = nib.load(os.path.join(func_path, dir_list[0]))

        self.mni_dim = function_map.get_fdata().shape
        self.affine = function_map.affine
        self.voxel_dims = [int(self.affine[0, 0])] * 3
        self.newsavepath = self.savepath

    def make_flat_gm_mask(self, tpm_path):
        """

        Parameters
        ----------
        tpm_path

        Returns
        -------

        """
        tpm = nib.load(tpm_path).get_fdata()
        gm = tpm[:, :, :, 0]

        affine = nib.load(tpm_path).affine
        if affine[0, 0] < 0:
            affine[0, 0] = -affine[0, 0]
            affine[0, 3] = -affine[0, 3]

        newaffine = rescale_affine(affine, self.voxel_dims)
        self.affine = newaffine

        gm = image.resample_img(nib.Nifti1Image(gm, affine),
                                target_affine=newaffine,
                                target_shape=self.mni_dim).get_fdata()

        mask = gm > self.grey_matter_thresh

        flat_mask = mask.flatten()
        self.flat_mask = np.nonzero(flat_mask)[0]
        nib.save(nib.Nifti1Image(mask.astype(float), affine), os.path.join(self.savepath, "mask.nii"))

        wm = tpm[:, :, :, 1]
        wm = image.resample_img(nib.Nifti1Image(wm, affine),
                                target_affine=newaffine,
                                target_shape=self.mni_dim).get_fdata()
        wm_mask = wm > self.white_matter_thresh
        nib.save(nib.Nifti1Image(wm_mask.astype(float), affine), os.path.join(self.savepath, "wm_mask.nii"))

        return mask.astype(int), wm_mask.astype(int), self.flat_mask, newaffine

    def load_data(self, func_path, flat_mask):
        """

        Parameters
        ----------
        func_path
        flat_mask

        Returns
        -------

        """
        dir_list = os.listdir(func_path)
        data = np.zeros([len(dir_list), len(flat_mask)])
        for i in range(len(dir_list)):
            term = dir_list[i]
            if self.z_thresh:
                masked_pos = np.zeros(self.mni_dim)
                function_map = nib.load(os.path.join(func_path, term)).get_fdata()
                masked_pos[function_map > self.z_thresh] = 1
                cc_pos, n_pos = ndi.label(masked_pos)
                inds, counts = np.unique(cc_pos, return_counts=True)
                inds_to_keep = inds[counts > 27]
                clean_masked_pos = np.zeros(self.mni_dim)
                for x in inds_to_keep:
                    if x != 0:
                        clean_masked_pos += (cc_pos == x).astype(int)

                masked_neg = np.zeros(self.mni_dim)
                masked_neg[function_map < -self.z_thresh] = 1
                cc_neg, n_neg = ndi.label(masked_neg)
                inds, counts = np.unique(cc_neg, return_counts=True)
                inds_to_keep = inds[counts > 27]
                clean_masked_neg = np.zeros(self.mni_dim)
                for x in inds_to_keep:
                    if x != 0:
                        clean_masked_neg += (cc_neg == x).astype(int)

                masked = np.zeros(self.mni_dim)
                masked[clean_masked_pos == 1] = 1
                masked[clean_masked_neg == 1] = -1
            else:
                masked = nib.load(os.path.join(func_path, term)).get_fdata()
            # This line loads the nifti, converts to numpy, flattens to vector and binary masks
            data[i, :] = np.take(masked.flatten(), flat_mask)
        print(f'Data loaded. Shape={data.shape}')
        return data.T

    def fit_clustering_model(self, data):
        """

        Parameters
        ----------
        data

        Returns
        -------

        """
        if self.loadpath:
            model = joblib.load(self.loadpath)
        else:
            if self.dimred:
                if self.dimred.lower() == 'pca':
                    data = PCA(n_components=self.dimred_components).fit_transform(data)
                    joblib.dump(data, os.path.join(self.savepath, f'pca_reduced_{int(self.dimred_components)}.joblib'))
                elif self.dimred.lower() == 'nmf':
                    data = NMF(n_components=self.dimred_components).fit_transform(data)
                    joblib.dump(data, os.path.join(self.savepath, f'nmf_reduced_{int(self.dimred_components)}.joblib'))
                model = AgglomerativeClustering(n_clusters=None, distance_threshold=self.dist_thresh).fit(data)
                joblib.dump(model, os.path.join(self.savepath,
                                                f'agg_cluster_model_dimred_{self.dimred}_{int(self.dimred_components)}_thresh_{self.dist_thresh}.joblib'))
            else:
                model = AgglomerativeClustering(n_clusters=None, distance_threshold=self.dist_thresh).fit(
                    data)
                joblib.dump(model, os.path.join(self.savepath,
                                                f'agg_cluster_model_dimred_{self.dimred}_{int(self.dimred_components)}_thresh_{self.dist_thresh}.joblib'))

        if len(model.labels_) != len(self.flat_mask):
            raise ValueError('Grey matter mask not consistent with pretrained model.')
        print('Clustering model ready.')
        return model

    def vis_term_embeddings(self):
        """

        Returns
        -------

        """
        func_names = os.listdir(self.func_path)

        encoder = NeuroQueryModel.from_data_dir(datasets.fetch_neuroquery_model())
        voc = list(np.asarray(encoder.full_vocabulary()))

        words_all = [name[:-7] for name in func_names]
        embeddings = np.zeros((len(words_all), self.emb.shape[1]))
        for i, word in enumerate(words_all):
            try:
                embeddings[i, :] = self.emb[voc.index(word.replace('_', ' ')), :]
            except Exception as err:
                print(err)
        nmf = NMF(n_components=40, random_state=42).fit_transform(embeddings)
        tsne = TSNE(n_components=2, random_state=42).fit_transform(nmf)
        fig1 = plt.figure(figsize=(16, 12))
        fig1.suptitle('t-SNE of NMF reduction to 40 components from 300-element term embeddings')
        plt.scatter(tsne[:, 0], tsne[:, 1])
        fig1.savefig(os.path.join(self.savepath, 'embeddings.png'))

        fig2 = plt.figure(figsize=(32, 24))
        fig2.suptitle('t-SNE of NMF reduction to 40 components from 300-element term embeddings')
        plt.scatter(tsne[:, 0], tsne[:, 1])
        for x, y, text in zip(tsne[:, 0], tsne[:, 1], words_all):
            plt.annotate(text, (x, y))
        fig2.savefig(os.path.join(self.savepath, 'embeddings_labelled.png'))
        plt.close('all')
        return tsne

    def process_dendrogram(self, model, **kwargs):
        """

        Parameters
        ----------
        model
        kwargs

        Returns
        -------

        """
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

        dend_dict = dendrogram(linkage_matrix, **kwargs)

        return dend_dict

    def prepare_gene_ex_data(self):
        """

        Returns
        -------

        """
        inv_affine = np.linalg.inv(self.affine)
        coverage = np.zeros(self.mni_dim)
        mni_all = pd.DataFrame()
        for i, path in enumerate(self.gene_paths):

            microarray_expression = np.array(pd.read_csv(os.path.join(path, "MicroarrayExpression.csv")))
            if i == 0: all_gene_expression = np.zeros(microarray_expression.shape[0])

            microarray_expression = np.array(pd.read_csv(os.path.join(path, "MicroarrayExpression.csv")))
            microarray_expression = microarray_expression[:, 1:]

            SampleAnnot = pd.read_csv(os.path.join(path, "SampleAnnot.csv"))
            default_mni = SampleAnnot[['mni_x', 'mni_y', 'mni_z']]
            for i in range(len(default_mni)):
                coords = np.array(default_mni.iloc[i])
                coords_for_transform = np.ones(4)
                coords_for_transform[:3] = coords
                vox = np.matmul(inv_affine, coords_for_transform)[:-1]
                SampleAnnot.loc[i, 'x_vox'] = vox[0]
                SampleAnnot.loc[i, 'y_vox'] = vox[1]
                SampleAnnot.loc[i, 'z_vox'] = vox[2]
                coverage[int(vox[0]), int(vox[1]), int(vox[2])] = 1

            mni_all = mni_all.append(SampleAnnot[['x_vox', 'y_vox', 'z_vox']])
            all_gene_expression = np.vstack([all_gene_expression, microarray_expression.T])
            print(f"Coverage: {int(np.sum(coverage))}")
        # nib.save(nib.Nifti1Image(coverage, affine), os.path.join(savepath, 'coverage_all.nii'))
        all_gene_expression = all_gene_expression[1:, :]
        return mni_all, all_gene_expression

    def prepare_cmap(self, n_clust, median_proportions=False):
        """

        Parameters
        ----------
        n_clust
        median_proportions

        Returns
        -------

        """
        try:
            cmap = cc.cm[self.colourmap]
            palette = cc.palette[self.colourmap]
            rgb_palette = [hex2rgb(hex) for hex in palette]
        except:
            cmap = matplotlib.cm.get_cmap(self.colourmap)
            try:
                rgb_palette = np.array(cmap.colors) * 255
            except:
                mycmap = mcolors.LinearSegmentedColormap('new_cmap', cmap._segmentdata, 256)
                palette = [mcolors.rgb2hex(mycmap(i)) for i in range(mycmap.N)]
                rgb_palette = [hex2rgb(hex) for hex in palette]

        if median_proportions is not False:
            scaled_proportions = np.round(255 * median_proportions).astype(int)
            rgb = []
            for i in scaled_proportions:
                rgb.append(rgb_palette[i])
        else:
            rgb = []
            for i in np.linspace(0, len(rgb_palette) - 1, n_clust):
                rgb.append(rgb_palette[int(i)])

        # Arrange the custom colourmap in formats appropriate for various purposes.
        # These are as an RGB array and as a hex colour-list (hex_colours).
        rgb = np.array(rgb)
        rgb_float = rgb / 255
        hex_colours = np.zeros(rgb_float.shape[0]).astype(str)
        for i in range(rgb_float.shape[0]):
            hex_colours[i] = '#%02x%02x%02x' % (
                int(np.round(rgb[i, 0])), int(np.round(rgb[i, 1])), int(np.round(rgb[i, 2])))
        self.cmap = mcolors.ListedColormap(rgb_float)

        return rgb, rgb_float, hex_colours, cmap

    def proportional_cmap(self, dend_dict):
        """

        Parameters
        ----------
        dend_dict

        Returns
        -------

        """
        leaves = dend_dict['leaves_color_list']
        borders = get_border_index(leaves)
        borders.append(len(leaves))
        n_clust = len(borders)
        median_voxels = []
        for i in range(n_clust - 1):
            median_voxels.append((borders[i] + borders[i + 1]) / 2)
        median_proportions = np.array(median_voxels) / len(leaves)
        return median_proportions

    def export_cmaps(self, rgb):
        """
        Uses custom colour map generated from make_cmap to export for
        external software use. Saves ParaView importable, SurfIce importable and
        MRIcroGL importable colourmaps. If surficepath and mricropath are defined,
        the colourmap will be saved directly into the software path.

        Parameters
        ----------
        rgb : np.array
            Discrete custom colour map. len(n_clust)x3 where each row is a cluster
            group showing re, green and blue values for the colour map which will be
            used for the dendrogram, t-SNE reduction, ParaView, Surf Ice and MRIcroGL.
            Each term is 0-255.
        thresh : int, float or None
            Distance threshold for agglomerative clustering. Will be used in filename
            for saving cmaps for import.
        surficepath : str or bool
            Either path to Surf_Ice install location for exporting the cmap directly
            to application, or False to just save the importable cmap to savepath.
        mricropath : str or bool
            Either path to MRIcroGL install location for exporting the cmap directly
            to application, or False to just save the importable cmap to savepath.
        savepath : str
            savepath for Paraview importable cmap and if surficepath and/or
            mricropath is False, these cmaps too.

        Returns
        -------
        None.

        """
        # Derive n_clust from existing arguments.
        n_clust = rgb.shape[0]

        # Create xml colourmap for export to ParaView.
        root = et.Element('ColorMaps')
        items = et.SubElement(root, 'ColorMap')
        items.set('name', f'paraview_{self.current_thresh}_{n_clust}')
        items.set('space', 'RGB')
        point_add = et.SubElement(items, 'Point')

        for point in range(rgb.shape[0]):
            point_add = et.SubElement(items, 'Point')
            point_add.set('x', str(point + 1))
            point_add.set('o', '1')
            point_add.set('r', str(rgb[point, 0] / 255))
            point_add.set('g', str(rgb[point, 1] / 255))
            point_add.set('b', str(rgb[point, 2] / 255))

        # Save xml to savepath for manual import to ParView GUI.
        with open(os.path.join(self.newsavepath, f'cust_cmap_paraview_{self.current_thresh}_{n_clust}.xml'), "w") as f:
            f.write(et.tostring(root, encoding='unicode', method='xml'))

        # SurfIce and MRIcroGL require specific intensity specification, so split
        # intensity from 0,255 by number of clusters
        div255 = np.linspace(0, 255, rgb.shape[0]).astype(int)
        # Save to generic savepath for manual importation into Surfice
        f = open(os.path.join(self.newsavepath, f'cust_cmap_surfice_{self.current_thresh}_{n_clust}.clut'), 'x')
        f = open(os.path.join(self.newsavepath, f'cust_cmap_surfice_{self.current_thresh}_{n_clust}.clut'), 'a')
        # Format of the colourmap file requires these lines
        f.write(f'[FLT]\nmin=0\nmax=0\n[INT]\nnumnodes={rgb.shape[0] + 1}\n[BYT]\n')
        f.write('nodeintensity0=0\n')
        for i in range(rgb.shape[0]):
            f.write(f'nodeintensity{i + 1} = {div255[i]}\n')
        f.write('[RGBA255]\n')
        f.write('nodergba0=0|0|0|0\n')
        for i in range(rgb.shape[0]):
            f.write(f'nodergba{i + 1}={rgb[i, 0]}|{rgb[i, 1]}|{rgb[i, 2]}|0\n')
        f.close()

        div255 = np.linspace(1, 255, rgb.shape[0]).astype(int)
        # Save to generic savepath for manual importation into Surfice
        f = open(os.path.join(self.newsavepath, f'cust_cmap_mricro_{self.current_thresh}_{n_clust}.clut'), 'x')
        f = open(os.path.join(self.newsavepath, f'cust_cmap_mricro_{self.current_thresh}_{n_clust}.clut'), 'a')
        # Format of the colourmap file requires these lines
        f.write(f'[FLT]\nmin=0\nmax=0\n[INT]\nnumnodes={rgb.shape[0]}\n[BYT]\n')
        for i in range(rgb.shape[0]):
            f.write(f'nodeintensity{i + 1}={div255[i]}\n')
        f.write('[RGBA255]\n')
        for i in range(rgb.shape[0]):
            f.write(f'nodergba{i + 1}={rgb[i, 0]}|{rgb[i, 1]}|{rgb[i, 2]}|64\n')
        f.close()

    def reparcellate(self, dend_dict):
        """

        Parameters
        ----------
        dend_dict

        Returns
        -------

        """
        borders = get_border_index(dend_dict['leaves_color_list'])
        leaf_groups = np.zeros(len(dend_dict['ivl']))
        for i in range(len(borders) - 1):
            leaf_groups[borders[i]:borders[i + 1]] = i + 1
        leaf_groups[borders[i + 1]:] = i + 2
        voxel_groups = np.zeros(len(dend_dict['ivl']))
        for i in range(len(voxel_groups)):
            voxel_groups[int(dend_dict['ivl'][i])] = leaf_groups[i]
        flat_recon = np.zeros(self.mni_dim).flatten()
        for i in range(len(voxel_groups)):
            flat_recon[self.flat_mask[i]] = voxel_groups[i]
        recon = flat_recon.reshape(self.mni_dim)
        return recon

    def save_mni(self, parcel, name, **kwargs):
        """

        Parameters
        ----------
        parcel
        name
        kwargs

        Returns
        -------

        """
        nib.save(nib.Nifti1Image(parcel, self.affine), os.path.join(self.newsavepath, f'{name}.nii'))

        try:
            cmap = kwargs['cmap']
        except:
            cmap = self.cmap

        fig = plt.figure(figsize=(9, 18))
        plotting.plot_roi(nib.Nifti1Image(parcel, self.affine), cmap=cmap, vmin=1, display_mode='mosaic', figure=fig)#, axes=(0, 1, 0.5, 0.2))
        fig.savefig(os.path.join(self.newsavepath, f'{name}_slices.png'))
        if self.target_mni_dim != self.mni_dim:
            target_affine = copy.deepcopy(self.affine)
            target_shape = self.target_mni_dim

            recon_upsampled = image.resample_img(nib.Nifti1Image(parcel, self.affine), interpolation='nearest',
                                                 target_affine=target_affine, target_shape=target_shape)
            nib.save(recon_upsampled, os.path.join(self.newsavepath, f'{name}_upsampled.nii'))

            fig = plt.figure(figsize=(18, 9))
            plotting.plot_roi(recon_upsampled, cmap=cmap, vmin=1, display_mode='mosaic', figure=fig)
            fig.savefig(os.path.join(self.newsavepath, f'{name}_slices_upsampled.png'))

    def save_mni_A4(self, parcel, name, **kwargs):
        """

        Parameters
        ----------
        parcel
        name
        kwargs

        Returns
        -------

        """
        nib.save(nib.Nifti1Image(parcel, self.affine), os.path.join(self.newsavepath, f'{name}.nii'))

        try:
            cmap = kwargs['cmap']
        except:
            cmap = self.cmap

        num_cuts = 5
        mosaic = plotting.plot_roi(nib.Nifti1Image(parcel, self.affine), cmap=cmap, vmin=1, display_mode='mosaic', cut_coords=num_cuts)
        coords = mosaic.cut_coords

        fig, ax = plt.subplots(nrows=num_cuts, ncols=3, figsize=(8, 12))#(8.27, 11.69))
        display_modes = ['x', 'y', 'z']
        for x in range(3):
            for y in range(num_cuts):
                display_mode = display_modes[x]
                mni_coord = float(coords[display_modes[x]][y])
                if x == 0: plot_x = 2
                elif x == 1: plot_x = 1
                elif x == 2: plot_x = 0
                plotting.plot_roi(nib.Nifti1Image(parcel, self.affine), cmap=cmap, vmin=1, display_mode=display_mode,
                                  cut_coords=[mni_coord], axes=ax[y, plot_x], annotate=False)
                ax[y, plot_x].annotate(f'${display_mode} = {int(mni_coord)}$',xy=(0,0))
                if y == 0 and (plot_x == 0 or plot_x == 1):
                    ax[y, plot_x].annotate('L',xy=(0.15, 1))
                    ax[y, plot_x].annotate('R',xy=(0.75, 1))

        fig.savefig(os.path.join(self.newsavepath, f'{name}_slices.png'), dpi=300)
        """
        if self.target_mni_dim != self.mni_dim:
            target_affine = copy.deepcopy(self.affine)
            target_shape = self.target_mni_dim

            recon_upsampled = image.resample_img(nib.Nifti1Image(parcel, self.affine), interpolation='nearest',
                                                 target_affine=target_affine, target_shape=target_shape)
            nib.save(recon_upsampled, os.path.join(self.newsavepath, f'{name}_upsampled.nii'))

            fig = plt.figure(figsize=(18, 9))
            plotting.plot_roi(recon_upsampled, cmap=cmap, vmin=1, display_mode='mosaic', figure=fig)
            fig.savefig(os.path.join(self.newsavepath, f'{name}_slices_upsampled.png'))
        """


    def reclassify_small_ccs(self, parcel):
        """

        Parameters
        ----------
        parcel

        Returns
        -------

        """
        if self.knn > 1:
            recon_smallremoved = copy.deepcopy(parcel)
            for i in range(1, self.current_n_clust + 1):
                region = parcel == i
                region_cc = ndi.label(region)
                for j in range(1, region_cc[1] + 1):
                    single_cc = region_cc[0] == j
                    if single_cc.sum() < 64:
                        recon_smallremoved[single_cc] = np.nan
            nan_list = np.where(np.isnan(recon_smallremoved))
            not_nan = np.array(np.where(recon_smallremoved > 0)).T
            for x, y, z in zip(nan_list[0], nan_list[1], nan_list[2]):
                dists = np.sqrt(np.sum(np.square((np.array([x, y, z]) - not_nan)), axis=1))
                nearest = np.argsort(dists)
                neighbour_values = np.zeros(self.knn)
                for k in range(self.knn):
                    neighbour_coord = not_nan[nearest[k], :]
                    neighbour_values[k] = parcel[neighbour_coord[0], neighbour_coord[1], neighbour_coord[2]]
                recon_smallremoved[x, y, z] = int(statistics.mode(neighbour_values))
            return recon_smallremoved
        else:
            print('knn < 1 so no reclassification.')
        return parcel

    def amplify_cortex(self, parcel):
        """

        Parameters
        ----------
        parcel

        Returns
        -------

        """
        recon_mask = parcel != 0
        surfice_mask = binary_dilation(recon_mask, iterations=3).astype(int)
        inmask = np.where(surfice_mask != 0)
        inrecon = np.array(np.where(parcel != 0)).T
        for x, y, z in zip(inmask[0], inmask[1], inmask[2]):
            if parcel[x, y, z] != 0:
                surfice_mask[x, y, z] = parcel[x, y, z]
            else:
                dists = np.sqrt(np.sum(np.square((np.array([x, y, z]) - inrecon)), axis=1))
                if self.knn > 1:
                    nearest = np.argsort(dists)
                    neighbour_values = np.zeros(self.knn)
                    for k in range(self.knn):
                        neighbour_coord = inrecon[nearest[k], :]
                        neighbour_values[k] = parcel[neighbour_coord[0], neighbour_coord[1], neighbour_coord[2]]
                    surfice_mask[x, y, z] = int(statistics.mode(neighbour_values))
                else:
                    nearest_coord = inrecon[np.argmin(dists), :]
                    nearest_val = parcel[nearest_coord[0], nearest_coord[1], nearest_coord[2]]
                    surfice_mask[x, y, z] = int(nearest_val)
        return surfice_mask

    def plot_dendrogram(self, dend_dict, rgb_float, rgb_list, title):
        """

        Parameters
        ----------
        dend_dict
        rgb_float
        rgb_list
        title

        Returns
        -------

        """
        # icoord reflects the x-axis values in dendrogram plotting.
        icoord = np.array(dend_dict['icoord'])
        # dcoord reflects the y-axis values in dendrogram plotting.
        dcoord = np.array(dend_dict['dcoord'])

        # Plot dendrogram
        fig, ax = plt.subplots(figsize=self.figsize)
        xmin, xmax = icoord.min(), icoord.max()
        ymin, ymax = dcoord.min(), dcoord.max()
        ax.set_ylim(ymin, ymax + 0.1 * abs(ymax))
        ax.set_xlim(xmin - 10, xmax + 0.1 * abs(xmax))

        borders = get_border_index(dend_dict['leaves_color_list'])
        n_clust = len(borders)

        if rgb_list is False:
            rgb_list = np.zeros([len(dend_dict['color_list']), 3])
            for i in range(n_clust - 1):
                rgb_list[borders[i]:borders[i + 1]] = rgb_float[i]
            rgb_list[borders[i + 1]:] = rgb_float[i + 1]
            for i, c in enumerate(dend_dict['color_list']):
                if c == 'C0':
                    rgb_list[i, :] = 0
        # Plot dendrogram. Note: rate-limiting step. Can take a matter of minutes
        for xs, ys, r, g, b in zip(icoord, dcoord, rgb_list[:, 0], rgb_list[:, 1], rgb_list[:, 2]):
            plt.plot(xs, ys, color=(r, g, b))
        orig_yticks = list(ax.get_yticks())
        ax.set_yticks(orig_yticks + [self.current_thresh])
        ax.set_yticklabels(orig_yticks + ['THRESHOLD'])
        ax.hlines(self.current_thresh, xmin - 10, xmax + 0.1 * abs(xmax), colors='k', linestyles='dotted')
        # Hide messy x-ticks.
        ax.set_xticks([])
        ax.get_xaxis().get_major_formatter().set_scientific(False)

        # Set labels
        ax.set_xlabel('Voxels')
        ax.set_ylabel('Distance')
        #ax.set_title(title + f"\nNumber of groups = {int(n_clust)}, Distance threshold = {int(self.current_thresh)}")
        fig.savefig(os.path.join(self.newsavepath, f'dendrogram_{int(self.current_thresh)}_{int(n_clust)}.png'))

    def luminance_subgroups(self, dend_dict, data, rgb_float):
        """

        Parameters
        ----------
        dend_dict
        data
        rgb_float

        Returns
        -------

        """
        borders = get_border_index(dend_dict['leaves_color_list'])
        borders.append(len(dend_dict['leaves_color_list']))
        new_allcolours_mni = np.zeros((data.shape[0], 3))
        new_allcolours_dend = np.zeros((data.shape[0], 3))

        ivl = list(np.array(dend_dict['ivl']).astype(int))
        n_luminance = 3
        for i in range(len(borders) - 1):
            voxels_in_group = np.array(dend_dict['ivl'][borders[i]:borders[i + 1]]).astype(int)
            raw_data_in_group = np.zeros((voxels_in_group.shape[0], data.shape[1]))
            for j in range(len(voxels_in_group)):
                raw_data_in_group[j, :] = data[voxels_in_group[j], :]
            base_colour = rgb_float[i, :]
            singlegroup_model = AgglomerativeClustering(n_clusters=None, distance_threshold=0,
                                                        compute_distances=True).fit(raw_data_in_group)
            singlegroup_dist = np.floor(singlegroup_model.distances_.max())
            count = 0
            n_subclust = 0
            while n_subclust != n_luminance:
                singlegroup_dend_dict = self.process_dendrogram(singlegroup_model,
                                                                color_threshold=singlegroup_dist,
                                                                no_plot=True)
                singlegroup_borders = get_border_index(singlegroup_dend_dict['leaves_color_list'])
                n_subclust = len(singlegroup_borders)

                if n_subclust < n_luminance:
                    singlegroup_dist -= 50
                elif n_subclust > n_luminance:
                    singlegroup_dist += 10
                count += 1
                if count > 100:
                    n_luminance = n_subclust
                print(n_subclust)

            limits_works = []
            for limit in [0.25, 0.125, 0]:  # [0.5, 0.4, 0.3, 0.2, 0.1, 0]:
                luminances = np.linspace(1 - limit, 1 + limit, n_luminance)
                works = 0
                for lum in luminances:
                    new_colour = lighten_color(base_colour, lum)
                    if new_colour[0] > 0 and new_colour[0] < 1:
                        if new_colour[1] > 0 and new_colour[1] < 1:
                            if new_colour[2] > 0 and new_colour[2] < 1:
                                works += 1
                if works == len(luminances):
                    limits_works.append(limit)
            limit_to_use = np.max(limits_works)
            luminances = np.linspace(1 - limit_to_use, 1 + limit_to_use, n_luminance)

            singlegroup_borders.append(len(singlegroup_dend_dict['ivl']))
            for j in range(len(singlegroup_borders) - 1):
                new_colour = lighten_color(base_colour, luminances[j])
                voxels = np.array(
                    singlegroup_dend_dict['ivl'][singlegroup_borders[j]:singlegroup_borders[j + 1]]).astype(int)
                for k, vox in enumerate(voxels):
                    new_allcolours_dend[borders[i] + singlegroup_borders[j] + k] = new_colour
                    new_allcolours_mni[ivl[borders[i] + singlegroup_borders[j] + k], :] = new_colour
        return new_allcolours_dend, new_allcolours_mni

    def raw_reparcellate(self, mni_list):
        """

        Parameters
        ----------
        mni_list

        Returns
        -------

        """
        recon = np.zeros(self.mni_dim).flatten()
        for i in range(len(mni_list)):
            recon[self.flat_mask[i]] = mni_list[i]
        return recon.reshape(self.mni_dim)

    def reparcellate_luminance_subgroups(self, new_allcolours_mni):
        """

        Parameters
        ----------
        new_allcolours_mni

        Returns
        -------

        """
        unique_colours = np.unique(new_allcolours_mni, axis=0)
        n_unique = unique_colours.shape[0]
        mni_unique = np.zeros(new_allcolours_mni.shape[0])
        for i in range(mni_unique.shape[0]):
            for j in range(n_unique):
                if new_allcolours_mni[i, 0] == unique_colours[j, 0]:
                    if new_allcolours_mni[i, 1] == unique_colours[j, 1]:
                        if new_allcolours_mni[i, 2] == unique_colours[j, 2]:
                            mni_unique[i] = j + 1

        return self.raw_reparcellate(mni_unique), unique_colours

    def archetypal_terms(self, mni):
        """
        Cross-reference reconstructed clustering back against the original dataset
        and make lists of the closest 'matches' between cluster group and function.
        Note: Long run-time. Will consider GPU optimisation.

        Parameters
        ----------
        func_path : str
            Path to directory containing original .nii(.gz) files, named by function.
        mni : np.array
            Reconstruction in MNI space matching the dimensions of the neuroimages in
            func_path with each cluster group represented by integer voxels 1:n_clust.
        flat_mask : np.array
            indices in flattened mni space of voxels within the binary mask to be
            considered. I.e. intracranial, grey matter only etc.
        save : bool
            joblib.dump the functional array? Highly recommended as runtime for this
            function is long.
        savepath : str
            Path to dump array if save == True.
        thresh : int, float or None
            Distance threshold for the filename only.
        n_extract : int
            Number of 'matches' in order to extract for each cluster group.

        Returns
        -------
        functions : str np.array
            Each column reflects a cluster group. The terms in each column are the
            closest matches, in order of matching.
        """
        # How many terms are there in the dataset to crossreference to?
        # E.g. NeuroQuery.
        func_names = os.listdir(self.func_path)
        num_functions = len(func_names)

        # Extract number of cluster groups from existing arguments.
        n_clust = (np.unique(mni.astype(int)) > 0).sum()

        # Loop through all clusters, computing the dot-product with each reference
        # neuroimage. Save all in memory in large preallocated array.
        similarities = np.zeros([num_functions, n_clust])
        roi_list = np.unique(mni.astype(int))
        for func_idx in range(num_functions):
            name = func_names[func_idx]
            function = np.take(nib.load(self.func_path + name).get_fdata().flatten(), self.flat_mask)
            for roi_ind in range(1, n_clust + 1):
                roi = roi_list[roi_ind]
                if roi < 1 or roi > 200:
                    # If there are more than 200 clusters, reconsider methods. This
                    # function takes a matter of hours to process for ~20 cluster groups.
                    pass
                else:
                    # Make binary mask of cluster group and then use consistent binary mask too.
                    roimask = mni == roi
                    roimask = np.take(roimask.flatten(), self.flat_mask)

                    # Dot-product the cluster binary mask with each function to
                    # generate a score representing similarity.
                    if self.similarity_measure.lower() == 'dot':
                        similarities[func_idx, roi_ind - 1] = np.dot(function, roimask)
                    elif self.similarity_measure.lower() == 'cosine':
                        similarities[func_idx, roi_ind - 1] = cosine_similarity(function.reshape(1, -1),
                                                                                roimask.reshape(1, -1))
                    else:
                        raise ValueError('Please provide input for similarity_measure')

        if self.term_exclusivity:
            similarities_exclusive = np.zeros(similarities.shape)
            for i in range(similarities.shape[0]):
                term_sims = similarities[i, :]
                term_sims_argmax = np.argmax(term_sims)
                similarities_exclusive[i, term_sims_argmax] = term_sims[term_sims_argmax]
            similarities = similarities_exclusive

        # Now extract n_extract of the closest matches for each cluster
        allfuncs = os.listdir(self.func_path)
        #functions = np.zeros([self.n_terms, n_clust]).astype(str)
        functions = np.zeros([len(allfuncs), n_clust]).astype(str)
        for i in range(n_clust):
            list_all = similarities[:, i]

            # This line sorts vector descending order and returns indices
            idx = (-list_all).argsort()#[:self.n_terms]
            for ix in range(len(idx)):
                # Assuming all functions have .nii.gz, :-7 strips.
                functions[ix, i] = allfuncs[idx[ix]][:-7]

        return functions

    def plot_terms(self, functions, hex_colours, fig_scale=(6, 4), colour_thresh=100):
        """
        After archetypcal_terms(), this function can be used to plot the matching terms
        with each cluster elegently, with the background reflecting the unified colourmap.

        Parameters
        ----------
        functions : str np.array
            Returned from archetypcal_terms(). Each column reflects a cluster group.
            The terms in each column are the closest matches, in order of matching.
        hex_colours : str vector as np.array
            Hexadecimal representation of the colour of each cluster group, to match
            that of the MNI reconstruction and cmaps exported to other software.
        thresh : int, float or None
            Distance threshold for the filename and titles only.
        savepath : str
            Path to save figure(s) to if save == True.
        fig_scale : tuple
            Size of each subplot.
        colour_thresh : int or float
            If the total RGB intensity of the text exceeds this value, the text will
            be black, otherwise will be white. To ensure sufficient contract and not
            black text on navy background etc.
        save : bool
            Save figure(s)?

        Returns
        -------
        None.

        """
        # Extract from existing arguments.
        n_clust = functions.shape[1]

        # Preallocate and load in the background colours of each cluster from the
        # input custom colour mapping (hex_colours).
        colour_functions = np.zeros(functions.shape[1]).astype(str)
        for i in range(colour_functions.shape[0]):
            colour_functions[i] = hex_colours[i]

        # Select number of pages, rows and columns depending on how many cluster
        # groups there are. Max per page here is 24 (4x6).
        if n_clust < 13:
            npages, nrows, ncols = 1, 3, 4
        elif n_clust < 16:
            npages, nrows, ncols = 1, 3, 5
        elif n_clust < 17:
            npages, nrows, ncols = 1, 4, 4
        elif n_clust < 19:
            npages, nrows, ncols = 1, 3, 6
        elif n_clust < 21:
            npages, nrows, ncols = 1, 4, 5
        elif n_clust < 25:
            npages, nrows, ncols = 1, 4, 6
        elif n_clust > 24:
            npages = 1 + n_clust // 24
            nrows, ncols = 4, 6

        # Compute total figsize from fig_scale and number of rows and cols.
        #terms_figsize = (ncols * fig_scale[1], nrows * fig_scale[0])
        terms_figsize = (10, 12)


        # Plot the terms as tables in MatPlotLib subplots.
        clus_group = 0
        for page in range(npages):
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=terms_figsize)#, gridspec_kw={'height_ratios': [2, 1, 2, 1]})
            # Set up title
            """
            if self.current_thresh is not None:
                title = 'NeuroQuery terms most associated with each clustered voxel' + \
                        ' grouping, colour-coded to match MNI reconstruction.\n' + \
                        f'Clustering distance threshold = {self.current_thresh}'
            else:
                title = 'NeuroQuery terms most associated with each clustered voxel' + \
                        ' grouping, colour-coded to match MNI reconstruction.\n' + \
                        f'A priori number of cluster groups = {n_clust}'
            if npages > 1:
                fig.suptitle(f'{title}\nPage {page + 1}', fontweight='bold', y=0.85)
            else:
                fig.suptitle(title, fontweight='bold', y=0.85)
            """

            # Loop through rows and columns. If there are more rows and columns than
            # cluster groups, then an error will not be raised but nothing will be plot,
            # while the axes removed maintaining a clean rest of page.
            for i in range(nrows):
                for j in range(ncols):
                    ax[i, j].axis('off')
                    if clus_group < n_clust:
                        terms = [term.title() for term in functions[:, clus_group]]
                        terms = [term.replace('_', ' ', 1) for term in terms]
                        terms = [term.replace('_', '\n') for term in terms]
                        terms = [term.replace(' ', '\n') if len(term) > 20 and '\n' not in term else term for term in terms]
                        terms = np.array(terms).reshape(-1, 1)
                        table = ax[i, j].table(cellText=terms,
                                               cellLoc='center',
                                               cellColours=np.array([colour_functions[clus_group]] * \
                                                                    functions.shape[0]).reshape(-1, 1))
                        table_props = table.properties()
                        table_cells = table_props['children']
                        for n, cell in enumerate(table_cells):
                            if '\n' in terms[n, 0]:
                                cell.set_height(2*cell.get_height())
                            hex_colour = colour_functions[clus_group].lstrip('#')
                            R, G, B = tuple(int(hex_colour[i:i + 2], 16) for i in (0, 2, 4))
                            if (R * 0.299 + G * 0.587 + B * 0.114) > colour_thresh:
                                cell.get_text().set_color('#000000')
                            else:
                                cell.get_text().set_color('#FFFFFF')
                        clus_group += 1
            fig.tight_layout(rect=(0, 0.1, 1, 6))
            #fig.tight_layout()
            fig.subplots_adjust(top=0.99)
            fig.savefig(
                os.path.join(
                    self.newsavepath,
                    f'sim_{self.similarity_measure}_exclusive_{self.term_exclusivity}_terms_page_{page + 1}_thresh_{self.current_thresh}_{n_clust}.png'),
                dpi=300)

    def identify_themes(self, parcel, hex_colours):
        """

        Parameters
        ----------
        parcel
        hex_colours

        Returns
        -------

        """
        measures = []
        if self.similarity_measure == 'dot':
            measures.append('dot')
        elif self.similarity_measure == 'cosine':
            measures.append('cosine')
        else:
            measures = ['dot', 'cosine']

        exclusivities = []
        if self.term_exclusivity is True:
            exclusivities.append(True)
        elif self.term_exclusivity is False:
            exclusivities.append(False)
        else:
            exclusivities = [True, False]

        for similarity_measure in measures:
            self.similarity_measure = similarity_measure
            for exclusive in exclusivities:
                self.term_exclusivity = exclusive

                funcs = self.archetypal_terms(parcel)
                print('Archetypal terms identified.')

                self.plot_terms(funcs[:self.n_terms, :], hex_colours)
                print('Archetypal terms exported.')

        return funcs

    def plot_embedding_coherence(self, function_lists, tsne, rgb_float):
        """

        Parameters
        ----------
        function_lists
        tsne
        rgb_float

        Returns
        -------

        """
        func_names = os.listdir(self.func_path)
        words_all = [name[:-7] for name in func_names]
        colours = np.zeros((len(words_all), 3))
        err_list = []
        for i in range(len(words_all)):
            word = words_all[i]
            try:
                inds = []
                for j in range(function_lists.shape[1]):
                    inds.append(list(function_lists[:, j]).index(word))
                colours[i, :] = rgb_float[np.argmin(inds)]
            except Exception as err:
                print(err)
                err_list.append(err)
            print(i)

        fig1 = plt.figure(figsize=self.figsize)
        #fig1.suptitle(
        #    f't-SNE of NMF reduction to 40 components from 300-element term embeddings\nNumber of functional groups: {self.current_n_clust}\nColoured by functional parcellation group of the activation map associated with the term.')
        plt.scatter(tsne[:, 0], tsne[:, 1], c=colours)
        fig1.savefig(os.path.join(self.newsavepath, 'word_embeddings_coloured.png'))

        fig2 = plt.figure(figsize=(32, 24))
        #fig2.suptitle(
        #    f't-SNE of NMF reduction to 40 components from 300-element term embeddings\nNumber of functional groups: {self.current_n_clust}\nColoured by functional parcellation group of the activation map associated with the term.')
        plt.scatter(tsne[:, 0], tsne[:, 1], c=colours)
        for x, y, text in zip(tsne[:, 0], tsne[:, 1], words_all):
            plt.annotate(text, (x, y))
        fig2.savefig(os.path.join(self.newsavepath, 'word_embeddings_coloured_labelled.png'))
        plt.close('all')

    def topic_summary(self, function_lists):
        """

        Parameters
        ----------
        function_lists

        Returns
        -------

        """
        encoder = NeuroQueryModel.from_data_dir(datasets.fetch_neuroquery_model())
        voc = list(np.asarray(encoder.full_vocabulary()))
        allresults = pd.DataFrame()
        topics = []
        for cutoff in np.arange(1, 51, 1):
            funcs_cutoff = function_lists[:cutoff, :]
            for col in range(funcs_cutoff.shape[1]):
                lst = funcs_cutoff[:, col]
                weights = np.arange(len(lst), 0, -1)
                weights_mag = np.sum(weights)
                avg = np.zeros(300)
                for i, term in enumerate(lst):
                    try:
                        avg += self.emb[voc.index(term.replace('_', ' ')), :] * weights[i]
                    except:  # Exception as err:
                        pass  # print(err)
                avg /= weights_mag

                dists = np.sqrt(np.sum(np.square(avg - self.emb), axis=1))
                dist_topic = voc[np.argmin(dists)]
                topics.append(dist_topic)
                print(f"Dists: {dist_topic}")

                allresults = allresults.append({'cutoff': cutoff,
                                                'roi': col,
                                                'dist': dist_topic},
                                               ignore_index=True)
        return topics, allresults

    def automate_topic_extraction(self, allresults, group):
        """

        Parameters
        ----------
        allresults
        group

        Returns
        -------

        """
        topics = allresults.loc[allresults['roi'] == group - 1]
        topics = topics.reset_index()
        for i in range(len(topics)):
            topics.loc[i, 'weight'] = 1 / (i + 1)
        s = pd.Series(list(topics['dist']))
        onehot = pd.get_dummies(s)
        weighted_onehot = np.dot(np.array(onehot).T, np.array(topics['weight']))
        topic_ind = np.argmax(weighted_onehot)
        return onehot.columns[topic_ind]

    def cluster_gene_ex_in_mask(self, mask, mni_all, all_gene_expression):
        """

        Parameters
        ----------
        mask
        mni_all
        all_gene_expression

        Returns
        -------

        """
        gene_index = []
        gene_index_mni = []
        for i in range(len(mni_all)):
            x, y, z = mni_all.iloc[i]
            if mask[int(x), int(y), int(z)]:
                gene_index.append(i)
                gene_index_mni.append([x, y, z])
        gene_data_in_roi = np.zeros((len(gene_index), all_gene_expression.shape[1]))
        for i in range(len(gene_index)):
            gene_data_in_roi[i, :] = all_gene_expression[gene_index[i], :]
        cluster = AgglomerativeClustering(n_clusters=2).fit(gene_data_in_roi)
        clustering = cluster.labels_ + 1
        recon = np.zeros(self.mni_dim)
        for i in range(len(clustering)):
            x, y, z = mni_all.iloc[gene_index[i]]
            recon[int(x), int(y), int(z)] = clustering[i]
        gene_data_in_roi = pd.DataFrame(gene_data_in_roi)
        gene_data_in_roi['label'] = clustering
        gene_index_mni = np.array(gene_index_mni)
        gene_data_in_roi["x"] = gene_index_mni[:, 0]
        gene_data_in_roi["y"] = gene_index_mni[:, 1]
        gene_data_in_roi["z"] = gene_index_mni[:, 2]
        return recon, gene_data_in_roi

    def gene_ex_nearest_neighbour_func(self, mask, gene_data_in_roi):
        """

        Parameters
        ----------
        mask
        gene_data_in_roi

        Returns
        -------

        """
        mni_roi = gene_data_in_roi[['x', 'y', 'z', 'label']]
        points = np.array(mni_roi[['x', 'y', 'z']])
        labels = np.array(mni_roi['label'])
        nearest_neighbour = np.zeros(self.mni_dim)
        for x in range(nearest_neighbour.shape[0]):
            for y in range(nearest_neighbour.shape[1]):
                for z in range(nearest_neighbour.shape[2]):
                    if mask[x, y, z] == 1:
                        vox = np.array([x, y, z])
                        dists = vox - points
                        euc_dists = np.sqrt(np.sum(dists ** 2, axis=1))
                        nearest = np.argmin(euc_dists)
                        nearest_neighbour[x, y, z] = labels[nearest]

        return nearest_neighbour

    def gaussian_classification(self, points_recon, mask, sd):
        """

        Parameters
        ----------
        points_recon
        mask
        sd

        Returns
        -------

        """
        n_splits = np.max(points_recon)
        mask_idx = np.nonzero(mask.flatten())[0]
        gaussed_voxels = np.zeros((len(mask_idx), int(n_splits)))
        classified_voxels = np.zeros(len(mask_idx))

        groups = []
        for i in range(1, int(n_splits + 1)):
            group = points_recon == i
            gaussed = ndi.gaussian_filter(group.astype(float), sd).flatten()
            for j in range(len(mask_idx)):
                gaussed_voxels[j, i - 1] = gaussed[mask_idx[j]]
            groups.append(np.array(np.where(group)).T)
        for i in range(len(mask_idx)):
            if gaussed_voxels[i, 0] > gaussed_voxels[i, 1]:
                classified_voxels[i] = 1
            elif gaussed_voxels[i, 0] < gaussed_voxels[i, 1]:
                classified_voxels[i] = 2
            else:
                vox = np.zeros(len(mask.flatten()))
                vox[mask_idx[i]] = 1
                vox = vox.reshape(mask.shape)
                coords = np.array(np.where(vox)).flatten()
                dists = []
                for g in range(len(groups)):
                    dists.append(np.sqrt(np.sum(np.square(coords - groups[g]), axis=1)))
                all_dists = np.sort(np.hstack(dists))
                points_1 = 0
                points_2 = 0
                for k in range(self.knn):
                    if all_dists[k] in dists[0] and all_dists[k] not in dists[1]:
                        points_1 += 1
                    elif all_dists[k] not in dists[0] and all_dists[k] in dists[1]:
                        points_2 += 1
                if points_1 > points_2:
                    classified_voxels[i] = 1
                elif points_2 > points_1:
                    classified_voxels[i] = 2
                else:
                    if np.sum(dists[0][:self.knn]) > np.sum(dists[1][:self.knn]):
                        classified_voxels[i] = 1
                    elif np.sum(dists[1][:self.knn]) > np.sum(dists[0][:self.knn]):
                        classified_voxels[i] = 2
                    else:
                        if np.mean(dists[0]) > np.mean(dists[1]):
                            classified_voxels[i] = 1
                        elif np.mean(dists[1]) > np.mean(dists[0]):
                            classified_voxels[i] = 2
                        else:
                            classified_voxels[i] = np.random.randint(1, 3)
        flat_recon = np.zeros(mask.size)
        for i in range(len(mask_idx)):
            flat_recon[mask_idx[i]] = classified_voxels[i]
        return flat_recon.reshape(mask.shape)


    def gen_roi_pairs(self, parcel, topics, vol_thresh=27):
        """
        Subdivide each cluster group into pairs that can be used to simulate similar
        deficits if a lesion affects one and/or both of these ROI. If there are two
        connected components, these will be returned as a pairing, if there is a
        single connected component or more than 2, the downstream hierarchical
        clustering pair will be returned. Saved as nii files with voxels labelled
        1 or 2.

        Parameters
        ----------
        mni : np.array
            Parcellation of functional loci. Each functional clustering group should
            be labelled with consecutive integers starting at 1.
        data : pd.DataFrame
            DataFrame of each voxels distribution. Likely a number of PCA components
            or similar dimensionality reduction method. Final column should be titled
            'Cluster', mapping the voxel to its cluster group. The length should be
            equal to that of a mask (e.g. intracranial), for which the mapping to
            flat MNI space is given in flat_mask.
        flat_mask : np.array
            List of indices in flat MNI space where the mask maps to.
        affine_matrix : np.array, optional
            affine for saving .nii files. Can find using nib.load(img).affine.
            The default is None.
        save : bool, optional
            Save ROI pairs as .nii? The default is False.
        savepath : str, optional
            Path to save ROI pair .nii if save == True. The default is False.

        Returns
        -------
        None.

        """
        # Initialise variables not defined

        dir_list = os.listdir(self.func_path)
        data = np.zeros([len(dir_list), len(self.flat_mask)])
        for i in range(len(dir_list)):
            term = dir_list[i]
            if self.z_thresh:
                masked = nib.load(os.path.join(self.func_path, term)).get_fdata() > self.z_thresh
            else:
                masked = nib.load(os.path.join(self.func_path, term)).get_fdata()
            # This line loads the nifti, converts to numpy, flattens to vector and binary masks
            data[i, :] = np.take(masked.flatten(), self.flat_mask)
        data = data.T
        data = pd.DataFrame(data)

        flat_parcel = parcel.flatten()
        flat_masked_parcel = np.zeros(self.flat_mask.shape)
        for i in range(len(self.flat_mask)):
            flat_masked_parcel[i] = flat_parcel[self.flat_mask[i]]
        data['Cluster'] = flat_masked_parcel
        n_clust = (np.unique(parcel.astype(int)) > 0).sum()

        self.newsavepath = os.path.join(self.newsavepath, 'functional_separation_hierarchy/')
        try:
            os.makedirs(self.newsavepath)
        except Exception as err:
            print(err)

        # Loop through cluster groups
        for i in range(1, n_clust + 1):
            # Make binary representation of cluster group alone
            group = parcel == i

            # Compute number of connected components
            cc = label(group, connectivity=2, return_num=True)
            voxels = np.unique(cc[0], return_counts=True)[1][1:]
            large_enough = voxels > vol_thresh

            # If there are only 2 connected components, use each one as its own
            # ROI within the paired deficit associated loci.
            if (cc[1] == 2) and (False not in large_enough):
                pair = cc[0]

            # If there is a single connected component, or more than two, take the
            # downstream hierarchical clustering pair.
            else:
                # Extract voxels (after mask) labeled as the cluster group
                roi_pair = data.loc[data['Cluster'] == i].drop(columns=['Cluster'])

                # Refit the agglomerative clustering process. Since this is a
                # 'bottom-up' algorithm, it should be the same as its extraction
                # from the primary dendrogram
                clus = AgglomerativeClustering(n_clusters=2).fit(np.array(roi_pair))

                # Relabel each voxel within the extracted cluster with 1 or 2
                # relating to the deficit simulation loci.
                roi_single = clus.labels_ + 1

                # Using the index within the mask and that index within flattened
                # MNI space, reconstruct the MNI representation of the lone cluster
                # group, with subclustering pair labelled as 1 and 2.
                mni_flat = np.zeros(np.size(parcel))
                for l, v in zip(roi_single, roi_pair.index):
                    mni_flat[self.flat_mask[v]] = l

                # Reshape back to 3D MNI space.
                pair = mni_flat.reshape(parcel.shape)

            number = '0' + str(i) if i < 10 else str(i)
            #nib.save(nib.Nifti1Image(pair, self.affine), os.path.join(pairpath, f'{number}_{topics[i]}.nii'))

            self.save_mni_A4(pair, f'{number}_{topics[i-1]}', cmap='cividis_r')


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
