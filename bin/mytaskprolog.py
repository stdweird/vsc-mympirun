#!/usr/bin/env python
#
# Copyright 2009-2019 Ghent University
#
# This file is part of vsc-mympirun,
# originally created by the HPC team of Ghent University (http://ugent.be/hpc/en),
# with support of Ghent University (http://ugent.be/hpc),
# the Flemish Supercomputer Centre (VSC) (https://www.vscentrum.be),
# the Flemish Research Foundation (FWO) (http://www.fwo.be/en)
# and the Department of Economy, Science and Innovation (EWI) (http://www.ewi-vlaanderen.be/en).
#
# https://github.com/hpcugent/vsc-mympirun
#
# vsc-mympirun is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation v2.
#
# vsc-mympirun is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with vsc-mympirun.  If not, see <http://www.gnu.org/licenses/>.
#
"""
As part of srun task prolog
* Start MPS server/controller when CUDA_MPS_ACTIVE_THREAD_PERCENTAGE is set
* Generate single CUDA_VISIBLE_DEVICES to MYTASKPROLOG_ONE_GPU (if set)
  This is best-effort guess work to work around broken --gpus-per-task=1 --gpu_bind=closest
"""

from __future__ import print_function

import os
from vsc.utils.run import run

# read from environment, mainly for unittests
MPS_CONTROL = os.environ.get('MYTASKPROLOG_MPS_CONTROL', "/usr/bin/nvidia-cuda-mps-control")

def export(key, value):
    """print export key=value, which is picked up by the task prolog"""
    os.environ[key] = value
    print("export %s=%s" % (key, value))


def setup_mps():
    """Setup MPS controller/server and export some variables"""
    # TMPDIR should be per job
    mpsdir = os.path.join(os.environ['TMPDIR'], 'mps', os.environ['CUDA_VISIBLE_DEVICES'])
    try:
        os.makedirs(mpsdir)
    except Exception as _:
        # who cares, either it already exists and all is fine, or it doesn't and this is beyond saving
        pass

    # set the mpsdir per (list of) CUDA_VISIBLE_DEVICES
    #   then we have a mps-controller/server per set of closest gpus
    #      or single gpus if they fix the bug
    export('CUDA_MPS_PIPE_DIRECTORY', mpsdir)
    export('CUDA_MPS_LOG_DIRECTORY', mpsdir)

    # the daemon will figure out the race conditions
    run([MPS_CONTROL, "-d"])
    # do not start single server
    # if you run this with gpu-bind=closest and with or without working gpus-per-tasks
    #   you get all closest gpus; so let the server decide where to run what
    # server(s) will be started by first client trying to do something
    #   (as of volta, clients go to gpu directly, so no need for one per gpu)

    # there's also no need to unset the CUDA_VISIBLE_DEVICES, the numbering is correct
    #   even after MPS takes over (it's already relative to the constrained devices)


def main():
    if 'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE' in os.environ:
        setup_mps()


if __name__ == '__main__':
    main()
