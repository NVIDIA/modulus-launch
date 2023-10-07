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


import torch
import numpy as np
import os

@modulus.sym.main(config_path="./conf/", config_name="config")
def run(cfg: ModulusConfig) -> None:

    if DistributedManager().distributed:
        print("Multi-GPU currently not supported for this example. Exiting.")
        return

    input_keys = [Key("wavefield_in",size=21),]
    output_keys = [Key("wavefield_sol"),] 

    train_path = to_absolute_path(
        "./train_sets/data_scale_train.hdf5"
    )
    test_path = to_absolute_path(
        "./train_sets/data_scale_test.hdf5"
    )
    # make datasets
    train_dataset = HDF5GridDataset(
        train_path, invar_keys=["wavefield_in"], outvar_keys=["wavefield_sol"], n_examples=30
    )
    test_dataset = HDF5GridDataset(
        test_path, invar_keys=["wavefield_in"], outvar_keys=["wavefield_sol"], n_examples=2
    )

    # make list of nodes to unroll graph on
    decoder_net = instantiate_arch(
            cfg=cfg.arch.decoder,
            output_keys=output_keys,
            )
    fno = instantiate_arch(
            cfg=cfg.arch.fno,
            input_keys=input_keys,
            decoder_net=decoder_net,
            )

    nodes = [fno.make_node("fno")]

    # make domain
    domain = Domain()
    # add constraints to domain
    supervised = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset,
        batch_size=cfg.batch_size.grid,
        num_workers=4,  # number of parallel data loaders
    )
    domain.add_constraint(supervised, "supervised")
    # add validator
    val = GridValidator(
    nodes,
    dataset=test_dataset,
    batch_size=cfg.batch_size.validation,
    plotter=GridValidatorPlotter(n_examples=2),
    )
    domain.add_validator(val, "test")

    # make solver
    slv = Solver(cfg, domain)
    print("solver MOdel")
    print(slv.load_model)
    bsize=1

    # start solver
    slv.solve()

if __name__ == "__main__":
    run()

