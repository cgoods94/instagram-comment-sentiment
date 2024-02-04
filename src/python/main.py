import os

if ~(os.getcwd().endswith("src/python/")):
    os.chdir("src/python/")

from examples import taylor_swift_superbowl

taylor_sb_df = taylor_swift_superbowl()
