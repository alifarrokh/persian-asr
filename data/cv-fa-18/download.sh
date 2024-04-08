# Download cv-corpus-18.0-2024-06-14-fa.tar.gz from Mozilla Common Voice
if [ ! -d dataset ]; then
    if [ ! -f cv-corpus-18.0-2024-06-14-fa.tar.gz ]; then
        echo "Please download Common Voice Fa 18.0" && exit 1
    fi
    rm -rf cv-corpus-18.0-2024-06-14
    tar -xzf cv-corpus-18.0-2024-06-14-fa.tar.gz
    mv cv-corpus-18.0-2024-06-14/fa dataset
    rmdir cv-corpus-18.0-2024-06-14
fi
