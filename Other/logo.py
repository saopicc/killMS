import getpass
import os
from Other import ModColor

import subprocess


def print_logo():
    # cwd=os.getcwd()
    # KILLMS_DIR="%s/killMS2"%os.environ["KILLMS_DIR"]
    # os.chdir(KILLMS_DIR)
    # version = subprocess.check_output(["git", "describe"])
    # os.chdir(cwd)
    os.system('clear')
                                                       


    print """       __        _   __   __   ____    ____   ______     """
    print """      [  |  _   (_) [  | [  | |_   \  /   _|.' ____ \    """
    print """       | | / ]  __   | |  | |   |   \/   |  | (___ \_|   """
    print """       | '' <  [  |  | |  | |   | |\  /| |   _.____`.    """
    print """       | |`\ \  | |  | |  | |  _| |_\/_| |_ | \____) |   """
    print """      [__|  \_][___][___][___]|_____||_____| \______.'   """
    #print """             This is version : %s""" %ModColor.Str(version)
    print """                                                        """

