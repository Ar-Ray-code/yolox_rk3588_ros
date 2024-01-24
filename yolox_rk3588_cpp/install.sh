SCRIPT_DIR=$(cd $(dirname $0); pwd)
sudo cp $SCRIPT_DIR/lib/$(uname)/$(arch)/* /usr/local/lib/