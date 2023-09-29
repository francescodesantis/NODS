# %%
import h5py
import json
import numpy as np
import dill

# %%
with open("./demo_cerebellum.json", "r") as json_file:
    net_config = json.load(json_file)
cell_types = list(net_config["cell_types"].keys())


def get_positions(filename="300x_200z_claudia_dcn_test_3.hdf5"):
    with h5py.File(filename, "r") as f:
        cell_types_bsb = [
            "golgi_cell",
            "purkinje_cell",
            "mossy_fibers",
            "basket_cell",
            "stellate_cell",
            "glomerulus",
            "granule_cell",
            "dcn_cell_glut_large",
            "dcn_cell_GABA",
            "dcn_cell_Gly-I",
            "io_cell",
        ]
        neuronal_populations = {cell_name: {} for cell_name in cell_types_bsb}
        for cell_name in cell_types_bsb:
            ids = f["cells"]["placement"][cell_name]["identifiers"]
            neuronal_populations[cell_name]["cell_0_id"] = ids[0]
            neuronal_populations[cell_name]["numerosity"] = ids[1]

            try:
                neuronal_populations[cell_name]["cell_pos"] = f["cells"]["placement"][
                    cell_name
                ]["positions"][:]
            except Exception as e:
                print(f"{str(cell_name)} population has not been positioned")
                neuronal_populations[cell_name]["cell_pos"] = []
    return neuronal_populations


def get_connections(filename="300x_200z_claudia_dcn_test_3.hdf5"):
    with h5py.File(filename, "r") as f:
        connection_models_bsb = list(f["cells"]["connections"].keys())
        connectivity_matrices = {
            connection: {"id_pre": [], "id_post": []}
            for connection in connection_models_bsb
        }
        for connection_model in connection_models_bsb:
            connectivity_matrices[connection_model]["id_pre"] = list(
                f["cells"]["connections"][connection_model][:, 0].astype(int)
            )
            connectivity_matrices[connection_model]["id_post"] = list(
                f["cells"]["connections"][connection_model][:, 1].astype(int)
            )
    return connectivity_matrices


if __name__ == "__main__":
    hdf5_file = "cerebellum_330x_200z.hdf5"
    filename = "./demo_cerebellum_data/" + hdf5_file

    neuronal_populations = get_positions(filename)
    #NA MERDA: ID GRANULE E ID DCN GLUT SI SOVREPPONGONO
    cell_name='dcn_cell_glut_large'
    neuronal_populations[cell_name]["numerosity"]=neuronal_populations[cell_name]["numerosity"]-1
    neuronal_populations[cell_name]["cell_0_id"] = 31683
    network_geom_file = "geom_" + hdf5_file
    dill.dump(neuronal_populations, open(network_geom_file, "wb"))

    connectivity_matrices = get_connections(filename)
    for connection_model in list(connectivity_matrices.keys()):
        connectivity_matrices[connection_model]["id_pre"] = connectivity_matrices[connection_model]["id_pre"][:]+np.ones(len(connectivity_matrices[connection_model]["id_pre"][:]),dtype=int)
        connectivity_matrices[connection_model]["id_post"] = connectivity_matrices[connection_model]["id_post"][:]+np.ones(len(connectivity_matrices[connection_model]["id_post"][:]),dtype=int)
        print(connection_model,np.min(connectivity_matrices[connection_model]["id_pre"][:]))
    network_connectivity_file = "conn_" + hdf5_file
    dill.dump(connectivity_matrices, open(network_connectivity_file, "wb"))
