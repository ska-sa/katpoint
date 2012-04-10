svn co https://katfs.kat.ac.za/svnDS/code/tools
tools/kat_build_virtualenv.py -s checkout/system-requirements.txt -t venv
. venv/bin/activate
# Install other svn dependencies
# pip install svn+https://katfs.kat.ac.za/svnDS/code/katcp-python/trunk@${SVN_REVISION}

# Install self
pip install ./checkout