from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import torch
import set_pyreg_paths

import pyreg.simple_interface as SI
import pyreg.example_generation as EG
import pyreg.module_parameters as pars

import pyreg.similarity_measure_factory as py_smf

import pyreg.fileio as py_fio

params = pars.ParameterDict()
test_data = '/playpen/xhs400/Research/data/data_for_pregis_net/atlas_folder/atlas.nii.gz'
params.load_JSON('/playpen/xhs400/Research/PycharmProjects/pregis_net/main/settings/mermaid_config.json')
image_io = py_fio.ImageIO()
I, hdr, spacing, _ = image_io.read_batch_to_nc_format(test_data)
lncc = py_smf.CustLNCCSimilarity(spacing, params['model']['registration_model'])
#lncc = py_smf.NCCSimilarity(spacing,params)
print(I.shape)
sim = lncc.compute_similarity_multiNC(torch.from_numpy(I).cuda(), torch.from_numpy(I).cuda())
print(sim)
