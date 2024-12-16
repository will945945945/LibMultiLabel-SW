import libmultilabel.linear as linear
import numpy as np
import scipy.sparse as sparse
import argparse
import pickle
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--modelname', type=str, default="")
parser.add_argument('--input_dir', type=str, default="")
parser.add_argument('--output_dir', type=str, default="")

ARGS = parser.parse_args()

if "selection" in ARGS.modelname.lower():
    with open(os.path.join( ARGS.input_dir, ARGS.modelname), "rb") as F:
        level_0_model, level_1_model, indices = pickle.load(F)

    level_0_model.weights = sparse.csr_matrix(level_0_model.weights)
    level_1_model.flat_model.weights = level_1_model.flat_model.weights.tocsr()

    with open(os.path.join( ARGS.output_dir, ARGS.modelname), "wb") as E:
        pickle.dump( (level_0_model, level_1_model, indices), E, protocol=5)

elif "forest" in ARGS.modelname.lower() and "replace" in ARGS.modelname.lower():
    with open(os.path.join( ARGS.input_dir, ARGS.modelname), "rb") as F:
        models, indices = pickle.load(F)

    #print(models)
    for idx in tqdm(range(len(models))):
        model, bla = models[idx]
        model.flat_model.weights = model.flat_model.weights.tocsr()
        models[idx] = (model, bla)
    
    with open(os.path.join( ARGS.output_dir, ARGS.modelname), "wb") as E:
        pickle.dump((models, indices), E, protocol=5)

else:
    with open(os.path.join( ARGS.input_dir, ARGS.modelname), "rb") as F:
        model, indices = pickle.load(F)

    model.flat_model.weights = model.flat_model.weights.tocsr()
    
    with open(os.path.join( ARGS.output_dir, ARGS.modelname), "wb") as E:
        pickle.dump((model, indices), E, protocol=5)
