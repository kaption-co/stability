#!/bin/bash

set -eo pipefail

##### Python + Pip + Poetry #####
# (echo ""; echo "##### Python + Pip + Poetry #####"; echo "";)
# sudo apt-get install -y python3-distutils
# sudo apt-get install -y python3-apt
# curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 -
# curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
# python3 get-pip.py
# rm get-pip.py
# source $HOME/.poetry/env

# latest python
# https://www.itsupportwale.com/blog/how-to-upgrade-to-python-3-10-on-ubuntu-18-04-and-20-04-lts/

##### Python + Pip + Poetry #####
(echo ""; echo "##### Python + Pip + Poetry #####"; echo "";)

sudo apt-get update
sudo apt-get install python3.9
sudo apt-get install python3.9-venv
python3.9 -m venv ./venv/main
source ./venv/main/bin/activate
sudo apt-get install -y python3-distutils
sudo apt-get install -y python3-apt
curl -sSL https://install.python-poetry.org | python3 -
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
rm get-pip.py
echo "" >> ~/.zshrc
echo "export PATH=$HOME/.local/bin:$PATH" >> ~/.zshrc
echo "" >> ~/.bashrc
echo "export PATH=$HOME/.local/bin:$PATH" >> ~/.bashrc
source ~/.zshrcx
source ~/.bashrc

pip3 install -r requirements.txt
conda install xformers -c xformers/label/dev
####################################################################################
##### Specify software and dependencies that are required for this project     #####
#####                                                                          #####
##### Note:                                                                    #####
##### The working directory is /home/brev/<PROJECT_FOLDER_NAME>. Execution of  #####
##### this file happens at this level.                                         #####
####################################################################################

##### Yarn #####
# (echo ""; echo "##### Yarn #####"; echo "";)
# curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add
# echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list
# sudo apt update
# sudo apt install -y yarn

##### Homebrew #####
# (echo ""; echo "##### Homebrew #####"; echo "";)
# curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh | bash -
# echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> /home/brev/.bash_profile
# echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> /home/brev/.zshrc
# eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"

##### Node v14.x + npm #####
# (echo ""; echo "##### Node v14.x + npm #####"; echo "";)
# sudo apt install ca-certificates
# curl -fsSL https://deb.nodesource.com/setup_14.x | sudo -E bash -
# sudo apt-get install -y nodejs
# # install npm packages globally without sudo
# if [ ! -d "${HOME}/.npm-packages" ]
# then
# mkdir "${HOME}/.npm-packages"
# printf "prefix=${HOME}/.npm-packages" >> $HOME/.npmrc
# cat <<EOF | tee -a ~/.bashrc | tee -a ~/.zshrc
# NPM_PACKAGES="\${HOME}/.npm-packages"
# NODE_PATH="\${NPM_PACKAGES}/lib/node_modules:\${NODE_PATH}"
# PATH="\${NPM_PACKAGES}/bin:\${PATH}"
# # Unset manpath so we can inherit from /etc/manpath via the `manpath`
# # command
# unset MANPATH # delete if you already modified MANPATH elsewhere in your config
# MANPATH="\${NPM_PACKAGES}/share/man:\$(manpath)"
# EOF
# fi

##### Custom commands #####
# (echo ""; echo "##### Custom commands #####"; echo "";)
# npm install
