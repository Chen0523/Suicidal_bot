import pandas as pd
import numpy as np
import json
import os
import torch
from datetime import datetime
from collections import defaultdict, OrderedDict

def load_model(output_path):
	cfgpath = os.path.join('output',output_path,'config.json')
	modelpath = os.path.join('output',output_path,'model.pt')

	with open(cfgpath, 'r') as f:
		config = json.load(f)

	model = SaptioTemporalNN(config['nt'], config['ns'], config['nz'], config['nhid'],
	                         config['nlayers'], config['nd'], config['dropout_f'], config['dropout_d']
	                         )

	state_dict = torch.load((modelpath), map_location=lambda storage, loc: storage)

	model.load_state_dict(state_dict)
	print('load model from ', output_path)

	return model
