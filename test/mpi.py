#
# Copyright 2012-2016 Ghent University
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
Tests for the vsc.mympirun.mpi.mpi module.

@author: Jeroen De Clerck
@author: Kenneth Hoste (HPC-UGent)
"""
from IPy import IP
import os
import pkgutil
import re
import stat
import string
import unittest
from vsc.utils.run import run_simple
from vsc.utils.missing import get_subclasses, nub

from vsc.mympirun.factory import getinstance
import vsc.mympirun.mpi.mpi as mpim
from vsc.mympirun.mpi.openmpi import OpenMPI
from vsc.mympirun.mpi.intelmpi import IntelMPI
from vsc.mympirun.option import MympirunOption
from vsc.mympirun.rm.local import Local

# we wish to use the mpirun we ship
os.environ["PATH"] = os.path.dirname(os.path.realpath(__file__)) + os.pathsep + os.environ["PATH"]


class TestMPI(unittest.TestCase):

    """tests for vsc.mympirun.mpi.mpi functions"""

    #######################
    ## General functions ##
    #######################

    def test_what_mpi(self):
        """test if what_mpi returns the correct mpi flavor"""
        scriptnames = ["ompirun", "mpirun", "impirun", "mympirun"]
        for scriptname in scriptnames:
            # if the scriptname is an executable located on this machine
            if mpim.which(scriptname):
                (returned_scriptname, mpi, found) = mpim.what_mpi(scriptname)
                print("what mpi returns: %s, %s, %s" % (returned_scriptname, mpi, found))
                # if an mpi implementation was found
                if mpi:
                    self.assertTrue(mpi in found,
                                    msg="returned mpi (%s) is not an element of found_mpi (%s)" % (mpi, found))
                    self.assertTrue(returned_scriptname == scriptname,
                                    msg="returned scriptname (%s) doesn't match actual scriptname (%s)" %
                                    (returned_scriptname, scriptname))
                else:
                    self.assertTrue(returned_scriptname.endswith("mpirun") or returned_scriptname is None,
                                    msg="no mpi found, scriptname should be the path to mpirun or None, but is %s" %
                                    returned_scriptname)

    def test_stripfake(self):
        """Test if stripfake actually removes the /bin/fake path in $PATH"""
        print("old path: %s" % os.environ["PATH"])
        mpim.stripfake()
        newpath = os.environ["PATH"]
        self.assertFalse(("bin/%s/mpirun" % mpim.FAKE_SUBDIRECTORY_NAME) in newpath, msg="the faked dir is still in $PATH")

    def test_which(self):
        """test if which returns a path that corresponds to unix which"""

        testnames = ["python", "head", "tail", "cat"]

        for scriptname in testnames:
            mpiwhich = mpim.which(scriptname)
            exitcode, unixwhich = run_simple("which " + scriptname)
            if exitcode > 0:
                raise Exception("Something went wrong while trying to run `which`: %s" % unixwhich)

            self.assertTrue(mpiwhich, msg="mpi which did not return anything, (unix which: %s" % unixwhich)
            self.assertEqual(mpiwhich, string.strip(unixwhich),
                             msg="the return values of unix which and which() aren't"" the same: %s != %s" %
                             (mpiwhich, string.strip(unixwhich)))

     ###################
     ## MPI functions ##
     ###################

    def test_options(self):
        """running mympirun with bad options"""
        optionparser = MympirunOption()
        optionparser.args = ['echo', 'foo', 'zever']
        # should not throw an error
        try:
            mpi_instance = getinstance(mpim.MPI, Local, optionparser)
            mpi_instance.main()
        except Exception:
            self.fail("mympirun raised an exception while running main()")

        optdict = mpi_instance.options.__dict__

        print("args given to mympirunoption: %s, instance options: %s, " % (optionparser.args, optdict))

        for opt in optionparser.args:
            self.assertFalse(opt in optdict)

    def test_is_mpirun_for(self):
        """test if _is_mpirun_for returns true when it is given the path of its executable"""
        optionparser = MympirunOption()

        mpidict = {
            IntelMPI: "/apps/gent/SL6/sandybridge/software/impi/3.2.2.006/bin64/mpirun",
            OpenMPI: "apps/gent/SL6/sandybridge/software/OpenMPI/1.8.8-GNU-4.9.3-2.25/bin/mpirun",
        }

        for key, val in mpidict.iteritems():
            instance = getinstance(key, Local, optionparser)

            print("mpiscriptname: %s, path: %s, instance mpirun for: %s" %
                  (instance._mpiscriptname_for, val,
                   instance._mpirun_for))
            self.assertTrue(instance._is_mpirun_for(val),
                            msg="mpi instance is not an MPI flavor defined by %s according to _is_mpirun_for, path: %s" %
                            (key, val))

    def test_set_omp_threads(self):
        """test if OMP_NUM_THREAD gets set correctly"""
        mpi_instance = getinstance(mpim.MPI, Local, MympirunOption())
        mpi_instance.set_omp_threads()
        self.assertTrue(getattr(mpi_instance.options, 'ompthreads') is not None, msg="ompthreads was not set")
        self.assertEqual(os.environ["OMP_NUM_THREADS"], getattr(mpi_instance.options, 'ompthreads', None),
                         msg="ompthreads has not been set in the environment variable OMP_NUM_THREADS")

    def test_set_netmask(self):
        """test if netmask matches the layout of an ip adress"""
        mpi_instance = getinstance(mpim.MPI, Local, MympirunOption())
        mpi_instance.set_netmask()
        # matches "IP address / netmask"
        reg = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
        print("netmask: %s" % mpi_instance.netmask)
        for substr in string.split(mpi_instance.netmask, sep=":"):
            try:
                IP(substr)
            except ValueError:
                self.fail()

    def test_select_device(self):
        """test if device and netmasktype are set and are picked from a list of options"""
        mpi_instance = getinstance(mpim.MPI, Local, MympirunOption())
        mpi_instance.select_device()
        self.assertTrue(mpi_instance.device and mpi_instance.device in mpi_instance.DEVICE_MPIDEVICE_MAP.values(),
                        msg="%s is not a valid device type, possible values: %s" %
                        (mpi_instance.device, mpi_instance.DEVICE_MPIDEVICE_MAP.values()))
        self.assertTrue(mpi_instance.netmasktype and mpi_instance.netmasktype in mpi_instance.NETMASK_TYPE_MAP.values(),
                        msg="%s is not a valid netmask type, possible values: %s" %
                        (mpi_instance.netmasktype, mpi_instance.NETMASK_TYPE_MAP.values()))

    def test_make_node_file(self):
        """test if the nodefile is made and if it contains the same amount of nodas as mpinodes"""
        mpi_instance = getinstance(mpim.MPI, Local, MympirunOption())
        mpi_instance.make_node_file()
        self.assertTrue(os.path.isfile(mpi_instance.mpiexec_node_filename), msg="the nodefile has not been created")

        # test if amount of lines in nodefile matches amount of nodes
        with open(mpi_instance.mpiexec_node_filename) as file:
            print("nodefile content: %s" % file)
            index = 0
            for index, _ in enumerate(file):
                pass
            self.assertEqual(len(mpi_instance.mpinodes), index+1,
                             msg="mpinodes doesn't match the amount of nodes in the nodefile")

    def test_make_mympirundir(self):
        """test if the mympirundir is made"""
        mpi_instance = getinstance(mpim.MPI, Local, MympirunOption())
        mpi_instance.make_mympirundir()
        self.assertTrue(mpi_instance.mympirundir and os.path.isdir(mpi_instance.mympirundir),
                        msg="mympirundir has not been set or has not been created")

    def test_make_mpdboot(self):
        """test if the mpdboot conffile is made and has the correct permissions"""
        mpi_instance = getinstance(mpim.MPI, Local, MympirunOption())
        mpi_instance.make_mpdboot()
        mpdconffn = os.path.expanduser('~/.mpd.conf')
        self.assertTrue(os.path.isfile(mpdconffn), msg="mpd.conf has not been created")
        perms = stat.S_IMODE(os.stat(mpdconffn).st_mode)
        self.assertEqual(perms, stat.S_IREAD, msg='permissions %0o for mpd.conf %s' % (perms, mpdconffn))

    def test_set_mpdboot_localhost_interface(self):
        """test if mpdboot_localhost_interface is set correctly"""
        mpi_instance = getinstance(mpim.MPI, Local, MympirunOption())
        mpi_instance.set_mpdboot_localhost_interface()
        (nodename, iface) = mpi_instance.mpdboot_localhost_interface
        self.assertTrue(mpi_instance.mpdboot_localhost_interface and nodename and iface)
        self.assertTrue((nodename, iface) in mpi_instance.get_localhosts(),
                        msg=("mpdboot_localhost_interface is not a result from get_localhosts, nodename: %s,"
                             " iface: %s, get_localhosts: %s"))

    def test_get_localhosts(self):
        """test if localhost returns a list containing that are sourced correctly"""
        mpi_instance = getinstance(mpim.MPI, Local, MympirunOption())
        res = mpi_instance.get_localhosts()
        _, out = run_simple("/sbin/ip -4 -o addr show")

        print("localhosts: %s" % res)

        for (nodename, interface) in res:
            self.assertTrue(nodename in mpi_instance.nodes,
                            msg="%s is not a node from the nodes list" % nodename)
            self.assertTrue(interface in out,
                            msg="%s can not be found in the output of `/sbin/ip -4 -o addr show`, output: %s" %
                            (interface, out))

    def test_set_mpiexec_global_options(self):
        """test if set_mpiexec_global_options merges os.environ and mpiexec_global_options"""
        mpi_instance = getinstance(mpim.MPI, Local, MympirunOption())
        mpi_instance.set_mpiexec_global_options()
        self.assertEqual(mpi_instance.mpiexec_global_options['MKL_NUM_THREADS'], "1",
                         msg="MKL_NUM_THREADS is not equal to 1")

        print("MODULE_ENVIRONMENT_VARIABLES: %s" % mpi_instance.MODULE_ENVIRONMENT_VARIABLES)

        if not mpi_instance.options.noenvmodules:
            for env_var in mpi_instance.MODULE_ENVIRONMENT_VARIABLES:
                self.assertEqual(env_var in os.environ, env_var in mpi_instance.mpiexec_global_options,
                                 msg=("%s is set in os.environ xor mpiexec_global_options, it should be set for both"
                                      " or set for neither") % env_var)

    def test_set_mpiexec_opts_from_env(self):
        """test if mpiexec_opts_from_env only contains environment variables that start with the given prefix"""
        mpi_instance = getinstance(mpim.MPI, Local, MympirunOption())
        mpi_instance.set_mpiexec_opts_from_env()
        prefixes = mpi_instance.OPTS_FROM_ENV_FLAVOR_PREFIX
        prefixes += mpi_instance.OPTS_FROM_ENV_BASE_PREFIX
        prefixes += mpi_instance.options.variablesprefix

        print("opts_from_env: %s" % mpi_instance.mpiexec_opts_from_env)
        for env_var in mpi_instance.mpiexec_opts_from_env:
            self.assertTrue(env_var.startswith(tuple(prefixes)),
                            msg="%s does not start with a correct prefix, prefixes %s" % (env_var, prefixes))
            self.assertTrue(env_var in os.environ, msg="%s is not in os.environ, while it should be" % env_var)

    def test_make_mpirun(self):
        """test if make_mpirun correctly builds the complete mpirun command"""
        inst = getinstance(mpim.MPI, Local, MympirunOption())
        inst.main()

        argspool = ['mpirun']
        argspool += inst.options.mpirunoptions if inst.options.mpirunoptions else []
        print("mpirunoptions: %s" % inst.options.mpirunoptions)
        argspool += inst.mpdboot_options
        print("mpdboot_options: %s" % inst.mpdboot_options)
        argspool += inst.mpiexec_options
        print("mpiexec_options: %s" % inst.mpiexec_options)
        argspool += inst.cmdargs
        print("cmdargs: %s" % inst.cmdargs)
        for arg in inst.mpirun_cmd:
            self.assertTrue(arg in argspool, msg="arg: %s, pool: %s" % (arg, argspool))

    def test_mympirun_aliases_setup(self):
        """Make sure that list of mympirun aliases included in setup.py is synced"""
        from setup import MYMPIRUN_ALIASES

        # make sure all modules in vsc.mympirun.mpi are imported
        for loader, modname, _ in pkgutil.walk_packages([os.path.dirname(mpim.__file__)]):
            loader.find_module(modname).load_module(modname)

        # determine actual list of mympirun aliases
        mympirun_aliases = ['myscoop']
        for mpiclass in get_subclasses(mpim.MPI):
            mympirun_aliases.extend(mpiclass._mpiscriptname_for)

        self.assertEqual(MYMPIRUN_ALIASES, nub(sorted(mympirun_aliases)))

    def test_fake_dirname(self):
        """Make sure dirname for 'fake' subdir is the same in both setup.py and vsc.mympirun.mpi.mpi"""
        from setup import FAKE_SUBDIRECTORY_NAME
        self.assertEqual(mpim.FAKE_SUBDIRECTORY_NAME, FAKE_SUBDIRECTORY_NAME)
