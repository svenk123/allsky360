git clone https://github.com/CESNET/GPUJPEG

cd GPUJPEG/

mkdir bld
cd bld
cmake ..   -DCMAKE_BUILD_TYPE=Release   -DGPUJPEG_CUDA_ARCHITECTURES="87"   -DCMAKE_INSTALL_PREFIX=/usr/local
make

sudo make install

cat > cat test_gpujpeg.c << EOF
#include <libgpujpeg/gpujpeg.h>
#include <stdio.h>

int main() {
    printf("GPUJPEG version: %d\n", gpujpeg_version());
    return 0;
}
EOF

sudo ldconfig
ldconfig -p | grep gpujpeg

gcc test_gpujpeg.c -I/usr/local/include -lgpujpeg -o test_gpujpeg

./test_gpujpeg 
GPUJPEG version: 6922
