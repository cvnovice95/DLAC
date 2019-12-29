#!/bin/bash
if [[ $1 == "" ]];then
	echo "Don't set workspace name!"
	exit
else
	echo "You set workspace name is $1"
fi
workspace="$(pwd)""/code/""$1"
local_bin="$(pwd)""/.local/bin"
echo "We will create workspace:$workspace"
echo "We will export path:$local_bin"
mkdir -p $workspace
echo "Are you sure? Please input answer [yes|no]"
read res
if [[ $res == 'yes' ]];then
	echo "you have selected 'yes'"
	echo "We will install the following tools:"
	echo "[1] zsh,ohmyzsh"
	echo "[2] neovim"
	echo "[3] tmux"
	echo "=>Install fonts-powerline, from: https://github.com/powerline/fonts"
	sudo apt-get install fonts-powerline
    echo "=>Install zsh"
	sudo apt-get install zsh
	echo "=>zsh version:""$(zsh --version)"
#	zsh
    export PATH=$local_bin:$PATH
	echo "=>Install neovim"
	sudo apt-get install software-properties-common
	sudo add-apt-repository ppa:deadsnakes/ppa
	sudo apt-get update
	sudo apt-get install python-dev python-pip python3-pip python3.7-dev 
	sudo apt-get install python3.7
	sudo add-apt-repository ppa:neovim-ppa/stable
	sudo apt-get update
	sudo apt-get install neovim
	sudo update-alternatives --install /usr/bin/vi vi /usr/bin/nvim 60
	sudo update-alternatives --config vi
	sudo update-alternatives --install /usr/bin/vim vim /usr/bin/nvim 60
	sudo update-alternatives --config vim
	sudo update-alternatives --install /usr/bin/editor editor /usr/bin/nvim 60
	sudo update-alternatives --config editor
	echo "=> You must use Python3 >=3.6"
	echo "sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.x 1"
	echo "sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2"
	echo "sudo update-alternatives --config python3"
    echo "==> Create Python Virtual Env"
	pip3 install neovim --user
	pip3 install pynvim --user
	pip3 install jedi   --user
	pip3 install yapf   --user
	pip3 install pylint --user
	pip3 install virtualenv --user
	cd $workspace
	echo $(pwd)
	virtualenv --no-site-packages  venv
	source ./venv/bin/activate
	pip3 install neovim 
	pip3 install pynvim 
	pip3 install jedi   
	pip3 install yapf   
	pip3 install pylint
	echo "==> Config Neovim"
	mkdir -p ~/.config/nvim
	touch -f ~/.config/nvim/init.vim
	curl -fLo ~/.local/share/nvim/site/autoload/plug.vim --create-dirs https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
	echo "call plug#begin('~/.local/share/nvim/plugged')
Plug 'Shougo/deoplete.nvim', { 'do': ':UpdateRemotePlugins' }
Plug 'zchee/deoplete-jedi'
Plug 'vim-airline/vim-airline'
Plug 'jiangmiao/auto-pairs'
Plug 'scrooloose/nerdcommenter'
Plug 'sbdchd/neoformat'
Plug 'davidhalter/jedi-vim'
Plug 'scrooloose/nerdtree'
Plug 'tmhedberg/SimpylFold'
Plug 'machakann/vim-highlightedyank'
Plug 'terryma/vim-multiple-cursors'
Plug 'neomake/neomake'
call plug#end()
let g:deoplete#enable_at_startup = 1
autocmd InsertLeave,CompleteDone * if pumvisible() == 0 | pclose | endif
inoremap <expr><tab> pumvisible() ? \"\<c-n>\" : \"\<tab>\"
let g:jedi#completions_enabled = 0
let g:jedi#use_splits_not_buffers = \"right\"
hi HighlightedyankRegion cterm=reverse gui=reverse
let g:neomake_python_enabled_makers = ['pylint']
" > ~/.config/nvim/init.vim	
	cat ~/.config/nvim/init.vim
	echo $(pwd)
	cd ~
	echo $(pwd)
	git clone https://github.com/gpakosz/.tmux.git
	ln -s -f .tmux/.tmux.conf
	cp .tmux/.tmux.conf.local .
	echo "=>Install Oh-my-zsh"
	sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
    echo "export PATH=$local_bin:$PATH" >> ~/.zshrc
	echo "=>In terminal, input nvim, then input :PlugInstall"
	echo "=>You can use (zsh) command into zsh env."
	echo "=>You can use (tmux new -s demo) command to use tmux tools"
	echo "=>finish!"
else
	echo "Good Bye"
fi



