import modulus
import modulus.sym
from modulus.sym.hydra import to_yaml, instantiate_arch, to_absolute_path
from modulus.sym.dataset import HDF5GridDataset
from modulus.sym.hydra.config import ModulusConfig
from modulus.sym.key import Key
from modulus.sym.domain.constraint import SupervisedGridConstraint
from modulus.sym.domain.validator import GridValidator
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.utils.io.plotter import GridValidatorPlotter
from modulus.sym.distributed.manager import DistributedManager
from utilities_jihyun import download_FNO_dataset, load_FNO_dataset
from modulus.sym.models.fno import FNOArch

import torch
import numpy as np
import os
import torch.nn as nn
import yaml,h5py

def read_yamlFile(fname):
    with open(fname, 'r') as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    return config_data

@modulus.sym.main(config_path="./conf/", config_name="config_FNO_custom_inv")
def run(cfg: ModulusConfig) -> None:

    input_keys = [Key("wavefield_in",size=21),]
    output_keys = [Key("wavefield_sol"),] 
    decoder_net = instantiate_arch(
            cfg=cfg.arch.decoder,
            output_keys=output_keys,
            )
    fno = instantiate_arch(
            cfg=cfg.arch.fno,
            input_keys=input_keys,
            decoder_net=decoder_net,
            )
    # load the trained model
    fno.load_state_dict(torch.load(to_absolute_path("./outputs/inv_FNO_brain2/fno.0.pth")))

    # setup device
    deviceNum = 0
    device = torch.device(f"cuda:{deviceNum}" if torch.cuda.is_available() else "cpu")
    fno_device = fno.to(device)

    # data you want to test 
    test_path = to_absolute_path(
        "./train_sets/data_scale_test.hdf5"
    )

    # Starting model
    base_path = to_absolute_path(
        "./train_sets/data_scale_base.hdf5"
    )
    with h5py.File(test_path, "r") as f:
        custom_key ='wavefield_sol'
        extracted_output = f[custom_key][()]  
        sol_numpy = extracted_output[1,:,:,:,:] 
    with h5py.File(base_path, "r") as f:
        custom_key ='wavefield_in'
        extracted_output = f[custom_key][()]  
        starting_input = extracted_output[0,:,:,:,:] 

    # start solver
    model= fno_device
    model.eval()

    test_wave_in = starting_input
    starting_dat= torch.from_numpy(test_wave_in).unsqueeze(0).to(device, torch.float32)

    ### initial starting velocity
    velSlice =nn.Parameter(starting_dat[:,20,:,:,:])  

    test_data_in ={
            "wavefield_in": starting_dat,
            }
    # inject starting velocity to input tensor
    test_data_in["wavefield_in"][:,20,:,:,:] = velSlice
    prediction =model(test_data_in)
    solTorch = torch.from_numpy(sol_numpy).to(device, torch.float32)

    # define loss function
    loss_fn = nn.MSELoss()
    # update loop.
    # W = W-(lr*W.grad.data)
    lr = 20000
    iterNum= 600
    f= open("loss_log.txt","w+") # located in working directory (e.g., ./outputs/<filename>/ )
    for i in range(iterNum):

    # inject updated elocity to input tensor
        test_data_in["wavefield_in"][:,20,:,:,:] = velSlice
        prediction =model(test_data_in)
        outTorch = prediction["wavefield_sol"]
        loss = loss_fn(outTorch,solTorch)

        if(i%10==0):
            print("iteration No. {}".format(i))
            print("loss: {}".format(loss))
        f.write("{}\n".format(loss))

        grad = torch.autograd.grad(loss, velSlice, torch.ones_like(loss))

        velSlice = velSlice - lr* grad[0][:,:,:,:]
        numpy_grad =grad[0].detach().cpu().numpy()

    # save gradient / updated velocity for every 100 step
        if(i%100==0):
            np.save('./grad/numpy_grad_'+str(i).zfill(3),numpy_grad)
            np.save('./grad/vel_'+str(i).zfill(3), velSlice.detach().cpu().numpy())

if __name__ == "__main__":
    run()

