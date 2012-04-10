svn co https://katfs.kat.ac.za/svnDS/code/tools
#note: no -s checkout/system-requirements.txt
tools/kat_build_virtualenv.py -t env
. venv/bin/activate
# Install other svn dependencies
# pip install svn+https://katfs.kat.ac.za/svnDS/code/katcp-python/trunk@${SVN_REVISION}

# Install self
pip install ./checkout

