from __future__ import print_function
import os
import os.path as osp
import numpy as np

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))


def get_output_dir(imdb_name, net_name=None,output_dir='output'):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """

    outdir = osp.abspath(osp.join(ROOT_DIR, output_dir, imdb_name))
    print(outdir)
    if net_name is not None:
        outdir = osp.join(outdir, net_name)

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir
