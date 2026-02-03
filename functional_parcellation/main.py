import os

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import scipy.ndimage as ndi
from scipy.stats import rankdata

from misc import func_parc, cmdargs_func_gene, get_border_index


def run():
    parcellate = func_parc(z_thresh,
                           dist_thresh,
                           grey_matter_thresh,
                           white_matter_thresh,
                           dimred,
                           dimred_components,
                           n_cluster_max,
                           knn,
                           similarity_type,
                           figsize,
                           colourmap,
                           local_or_hpc,
                           n_terms,
                           topn,
                           func_split,
                           loadpath,
                           notes,
                           target_mni_dim,
                           term_exclusivity)

    save_stem = ""
    func_path = ""
    tpm_path = ""
    gene_paths = []
    emb_path = ""

    parcellate.init_paths(save_stem, func_path, tpm_path, gene_paths, emb_path)

    mask, wm_mask, flat_mask, affine = parcellate.make_flat_gm_mask(tpm_path)
    print(f"Affine: {affine}")
    parcellate.save_mni_A4(wm_mask, 'white_matter_mask', cmap=colourmap)

    data = parcellate.load_data(func_path, flat_mask)

    model = parcellate.fit_clustering_model(data)

    tsne = parcellate.vis_term_embeddings()

    # Initialise values for the looping through dendrogram top down
    mni_all, all_gene_expression = parcellate.prepare_gene_ex_data()
    dist = 2460
    n_clusters_done = []
    dist_thresh_done = []

    dend_dict = parcellate.process_dendrogram(model, color_threshold=dist, no_plot=True)
    borders = get_border_index(dend_dict['leaves_color_list'])
    n_clust = len(borders)

    group_savepath = os.path.join(parcellate.savepath,
                                          f'n_clust_{int(n_clust)}_dist_{int(dist)}_cmap_{colourmap}')
    parcellate.newsavepath = group_savepath
    parcellate.current_thresh = dist
    parcellate.current_n_clust = n_clust
    os.makedirs(parcellate.newsavepath)
    rgb, rgb_float, hex_colours, cmap = parcellate.prepare_cmap(n_clust,
                                                                parcellate.proportional_cmap(dend_dict))
    parcellate.export_cmaps(rgb.astype(int))

    recon_orig = parcellate.reparcellate(dend_dict)
    parcellate.save_mni_A4(recon_orig, 'functional_parcellation')


    recon = parcellate.reclassify_small_ccs(recon_orig)
    parcellate.save_mni_A4(recon, 'functional_parcellation_clean')

    inflated = parcellate.amplify_cortex(recon)
    parcellate.save_mni_A4(inflated.astype(float), 'inflated')


    title = ""#"Agglomerative Clustering Dendrogram for Functional Data from NeuroQuery"
    parcellate.plot_dendrogram(dend_dict, rgb_float, False, title)

    function_lists = parcellate.identify_themes(recon, hex_colours)
    topics, allresults = parcellate.topic_summary(function_lists)
    parcellate.plot_embedding_coherence(function_lists, tsne, rgb_float)

    if func_split:
        parcellate.gen_roi_pairs(recon, topics)


    gene_points_savepath = os.path.join(group_savepath, 'gene_points')
    try: os.makedirs(gene_points_savepath)
    except Exception as err: print(err)

    nn_savepath = os.path.join(group_savepath, 'knn')
    try: os.makedirs(nn_savepath)
    except Exception as err: print(err)

    gauss_savepath = os.path.join(group_savepath, 'gauss')
    try: os.makedirs(gauss_savepath)
    except Exception as err: print(err)


    for group in range(1, int(np.max(recon) + 1)):
        topic = parcellate.automate_topic_extraction(allresults, group)

        mask = recon == group
        points_recon, gene_data_in_roi = parcellate.cluster_gene_ex_in_mask(mask, mni_all, all_gene_expression)
        parcellate.newsavepath = gene_points_savepath
        parcellate.save_mni_A4(points_recon, f'{group}_gene_points_{topic}', cmap='cividis_r')
        print('Gene expression points clustered. Nearest neighbour next...')

        nearest_neighbour = parcellate.gene_ex_nearest_neighbour_func(mask, gene_data_in_roi)
        parcellate.newsavepath = nn_savepath
        parcellate.save_mni_A4(nearest_neighbour, f'{group}_nearest_neighbour_{topic}', cmap='cividis_r')

        imbalance = []
        for sd in np.arange(0.25, 5.25, 0.25):
            try:
                gauss = parcellate.gaussian_classification(points_recon, mask, sd)
                vox_dist = np.unique(gauss, return_counts=True)[1][1:]
                cc_1 = np.max(ndi.label(gauss == 0)[0])
                cc_2 = np.max(ndi.label(gauss == 1)[0])
                imbalance.append([sd, np.abs(vox_dist[1] - vox_dist[0]), cc_1, cc_2])
            except Exception as err:
                print(err)
        imbalance = np.array(imbalance)
        vol_diff_rank = rankdata(imbalance[:,1], method='min')
        cc_1_rank = rankdata(imbalance[:,2], method='min')
        cc_2_rank = rankdata(imbalance[:,3], method='min')
        weighted_rank = [2 * cc_1_rank[i] + 2 * cc_2_rank[i] + vol_diff_rank[i] for i in range(imbalance.shape[0])]
        optimal_sd = imbalance[np.argmin(weighted_rank), 0]
        gauss = parcellate.gaussian_classification(points_recon, mask, optimal_sd)

        parcellate.newsavepath = gauss_savepath
        parcellate.save_mni_A4(gauss, f'{group}_gauss_sd_{optimal_sd}_{topic}', cmap='cividis_r')





    new_allcolours_dend, new_allcolours_mni = parcellate.luminance_subgroups(dend_dict, data, rgb_float)

    parcellate.newsavepath = os.path.join(group_savepath, f'luminance')
    try: os.makedirs(parcellate.newsavepath)
    except Exception as err: print(err)
    parcellate.plot_dendrogram(dend_dict, rgb_float, new_allcolours_dend, 'Luminance')

    recon_luminance, unique_colours = parcellate.reparcellate_luminance_subgroups(new_allcolours_mni)
    parcellate.save_mni_A4(recon_luminance, 'luminance')


    luminance_clean = parcellate.reclassify_small_ccs(recon_luminance)
    parcellate.save_mni_A4(luminance_clean, 'luminance_clean')

    inflated_luminance_clean = parcellate.amplify_cortex(luminance_clean)
    parcellate.save_mni_A4(inflated_luminance_clean.astype(float), 'inflated_luminance_clean')


    parcellate.export_cmaps((unique_colours * 255).astype(int))


    cmap_luminance = ListedColormap(unique_colours)
    parcellate.save_mni(recon_luminance.astype(float), name='subgroup_luminance', cmap=cmap_luminance)


    n_clusters_done.append(n_clust)
    dist_thresh_done.append(dist)

    plt.close('all')



if __name__ == "__main__":
    z_thresh, dist_thresh, grey_matter_thresh, white_matter_thresh, dimred, dimred_components, n_cluster_min, n_cluster_max, knn, similarity_type, figsize, colourmap, local_or_hpc, n_terms, topn, func_split, loadpath, notes, target_mni_dim, term_exclusivity = cmdargs_func_gene()
    local_or_hpc = 'local'
    loadpath = ""
    n_cluster_min = 16
    n_cluster_max = 16
    func_split = True
    run()
