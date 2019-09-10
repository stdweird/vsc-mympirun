#
# Copyright 2019-2019 Ghent University
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
PMI slurm tests
"""

import logging
logging.basicConfig(level=logging.DEBUG)

from pmi_utils import SLURM_2NODES, PMITest


class PMISimple(PMITest):
    def test_pmitest(self):
        """Test the PMITest class"""
        self.set_slurm_ompi4_ucx(SLURM_2NODES)
        notok = ['SLURM_CPUS_ON_NODE', 'SLURM_JOB_CPUS_PER_NODE', 'SLURM_MEM_PER_CPU',
                 'SLURM_NNODES', 'SLURM_NPROCS', 'SLURM_NTASKS', 'SLURM_JOB_NUM_NODES']
        ok = ['SLURM_JOB_ID', 'SLURM_JOB_NODELIST']
        self._check_vars(ok+notok)

        mpr = self.get_instance()

        self.assertEqual(mpr.LAUNCHER, 'srun', 'srun launcher')
        self.assertEqual(mpr.PMI[0].FLAVOUR, 'pmix', 'pmix flavour')

        pmicmd, run_function = mpr.pmicmd()

        # pmicmd unsets some of the slurm env
        self._check_vars(ok)
        self._check_vars(notok, missing=True)
