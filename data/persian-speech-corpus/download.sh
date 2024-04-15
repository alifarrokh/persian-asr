if [ ! -d dataset ]; then
    [ -f persian-speech-corpus.zip ] \
        || wget -q https://fa.persianspeechcorpus.com/persian-speech-corpus.zip \
        || (echo "Failed to download persian-speech-corpus.zip" && exit 1)
    mkdir dataset
    unzip -q -d dataset persian-speech-corpus.zip
else
    echo "The dataset already exists."
fi
