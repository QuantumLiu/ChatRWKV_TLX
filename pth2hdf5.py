import sys
import torch

import h5py

from tqdm import tqdm

def pth2hdf5(pth_path, h5_path):
    temp = torch.load(pth_path, map_location="cpu")
    print("Pytorch parameter names and parameter shapes:")
    for key in temp.keys():
        print(key, temp[key].shape)

    print("Saving parameters to .h5 file:")
    with h5py.File(h5_path, "w") as f:
        for key in tqdm(temp.keys()):
            data = temp[key]

            f.create_dataset(key, data=(data if data.dtype != torch.bfloat16 else data.to(torch.float32)))
            # print(def_rename_torch_key(key), def_torch_weight_reshape(temp[key]).shape)

if __name__ == "__main__":
    pth2hdf5(sys.argv[1], sys.argv[2])