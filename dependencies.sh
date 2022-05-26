
#SCSPKG
git clone https://github.com/scs-lab/scspkg.git
cd scspkg
bash install.sh
source ~/.bashrc

#SPACK
git clone https://github.com/spack/spack.git
. spack/share/spack/setup-env.sh
source ~/.bashrc

#MPICH (no fortran)
spack install mpich~fortran
spack load mpich

#Darshan
scspkg create darshan
cd `scspkg pkg-src darshan`
git clone https://github.com/darshan-hpc/darshan.git
cd darshan
git checkout 331118e742066ab93f8b27babbf683e94f2ca160
./prepare.sh
cd darshan-util
./configure --prefix=`scspkg pkg-root darshan` --enable-shared
make -j8
make install
cd ../darshan-runtime
./configure --prefix=`scspkg pkg-root darshan` --with-log-path-by-env=DARSHAN_LOG_PATH --with-jobid-env=DARSHAN_JOB_ID
make -j8
make install
module load darshan

#Redis
scspkg create redis
cd `scspkg pkg-src redis`
wget http://download.redis.io/redis-stable.tar.gz
tar xvzf redis-stable.tar.gz
cd redis-stable
make
make test
mkdir `scspkg pkg-root redis`/bin
cp src/redis-server `scspkg pkg-root redis`/bin
cp src/redis-cli `scspkg pkg-root redis`/bin
module load redis

#Luxio
scspkg create luxio
scspkg add-deps redis darshan-util darshan-runtime




#New repo: git clone https://github.com/darshan-hpc/darshan.git
#scspkg create darshan-util
#cd `scspkg pkg-src darshan-util`
#git clone https://xgitlab.cels.anl.gov/darshan/darshan.git
#cd darshan/darshan-util
#git checkout 932d69994064af315b736ab55f386cffd6289a15
#./configure --prefix=`scspkg pkg-root darshan-util` --enable-pydarshan --enable-shared
#make install
#cd pydarshan
#python3 -m pip install -r requirements.txt
#python3 setup.py sdist bdist_wheel
#python3 -m pip install dist/*.whl

#New repo: git clone https://github.com/darshan-hpc/darshan.git
#cd `scspkg pkg-src darshan-runtime`
#git clone https://xgitlab.cels.anl.gov/darshan/darshan.git
#cd darshan/darshan-runtime
#git checkout 932d69994064af315b736ab55f386cffd6289a15
#./configure --prefix=`scspkg pkg-root darshan-runtime` --with-log-path-by-env=DARSHAN_LOG_PATH --with-jobid-env=DARSHAN_JOB_ID
#make install
