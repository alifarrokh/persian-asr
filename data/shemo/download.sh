if [ ! -d dataset ]; then
    git_url="https://github.com/mansourehk/ShEMO.git"
    [ -d ShEMO ] || git clone $git_url || (echo "Failed to clone the repository." && exit 1)

    rm -rf dataset
    mkdir -p dataset/wavs

    unzip -q -d dataset/wavs/ ShEMO/female.zip || (echo "Failed to extract female.zip." && exit 1)
    unzip -q -d dataset/wavs/ ShEMO/male.zip || (echo "Failed to extract male.zip." && exit 1)
    unzip -q -d dataset/ ShEMO/transcript.zip || (echo "Failed to extract male.zip." && exit 1)
else
    echo "The dataset already exists."
fi
