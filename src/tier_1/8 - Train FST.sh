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


# deactivate the virtual environment to use conda
deactivate

# Intialize the Conda environment for SigMorphon
conda env create -f environment.yml
conda activate task1


# Overwrite the data directory for SigMorphon 2020 / FST 
rm -rf data
cp -r ../../../output/fst/ data 


# overwrite the local version of sweep with the one with edits so that it learns models for orders 1-3 and retains the symbol file rather than deleting it
cd baselines/fst
cp ../../../../../src/external/sweep .
# copy over the symbol file that covers all of the children
cp ../../../../../src/external/all_child_phones.sym .

# actually execute the aligment and modeling files
./sweep

# copy the resulting FST and symbol files back into the main codebase so that they can be called by existing code
fstprint checkpoints/chi-1.fst checkpoints/chi-1.txt
fstprint checkpoints/Alex-1.fst checkpoints/Alex-1.txt
fstprint checkpoints/Ethan-1.fst checkpoints/Ethan-1.txt
fstprint checkpoints/Lily-1.fst checkpoints/Lily-1.txt
fstprint checkpoints/Naima-1.fst checkpoints/Naima-1.txt
fstprint checkpoints/Violet-1.fst checkpoints/Violet-1.txt
fstprint checkpoints/William-1.fst checkpoints/William-1.txt


cp checkpoints/*-1.fst ../../../../../output/fst/
cp checkpoints/*-1.txt ../../../../../output/fst/


# deactivate the conda environment
conda deactivate

# restart the virtual environment for the rest of the code
cd ../../../../../
source child-directed-listening-env/bin/activate