#!/bin/bash

echo "🔧 Starting setup..." | tee ~/setup.log

# Install Python and pip
sudo apt update -y && sudo apt install -y python3-pip

# Upgrade pip and install packages
python3 -m pip install --upgrade pip
python3 -m pip install --break-system-packages -r ~/Heterogeneous-HEPPFL/requirements.txt
python3 -m pip install --break-system-packages numpy pandas websockets phe

# Update PATH to include ~/.local/bin and persist it
if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi
export PATH="$HOME/.local/bin:$PATH"

# Verify packages
echo "🔍 Verifying installation..." | tee -a ~/setup.log
python3 -c "import numpy, pandas, websockets, phe; print('✅ All packages installed successfully!')" 2>&1 | tee -a ~/setup.log

echo "🎉 Setup complete!" | tee -a ~/setup.log
