# install Pywrapfst
mkdir external
cd external
curl -O https://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.8.1.tar.gz
tar -xvzf openfst-1.8.1.tar.gz
cd openfst-1.8.1
./configure
make
sudo make install


# install BaumWelch
cd ../
curl -O https://www.opengrm.org/twiki/pub/GRM/BaumWelchDownload/baumwelch-0.3.6.tar.gz
tar -xvzf baumwelch-0.3.6.tar.gz
cd baumwelch-0.3.6
./configure
make
sudo make install


# install pynini
pip3 install Cython
python3 setup.py install


cd ../
curl -O https://www.opengrm.org/twiki/pub/GRM/PyniniDownload/pynini-2.1.4.tar.gz
tar -xvzf pynini-2.1.4.tar.gz
cd pynini-2.1.4


# Copy the SigMorphon 2020 Code
cd ../ 
git clone git@github.com:sigmorphon/2020.git
cd 2020/task1

# Intialize the environment for SigMorphon
conda env create -f environment.yml

# Overwrite the data directory for SigMorphon 2020 / FST 
rm -r data
cp -r ../../../fst_training data

# overwrite the local version of sweep with the one with edits so that it learns models for orders 1-3 and retains the symbol file rather than deleting it
cd baselines/fst
cp ../../../../sweep .

# actually execute the aligment and modeling files
./sweep

# copy the resulting FST and symbol files back into the main codebase so that they can be called by existing code
cp  chi_phones.sym ../../../../../fst/
cp checkpoints/chi-1.fst ../../../../../fst/