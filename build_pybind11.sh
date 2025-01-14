#git clone --recursive git@github.com:pybind/pybind11.git
mkdir -p pybind11/build
pushd pybind11/build
cmake .. -DCMAKE_INSTALL_PREFIX=../release -DCMAKE_BUILD_TYPE=Release -DPYBIND11_PYTHON_VERSION=3 -DPYBIND11_TEST=OFF
make install
popd
