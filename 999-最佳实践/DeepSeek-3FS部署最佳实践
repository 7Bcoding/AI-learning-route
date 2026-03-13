# 基于火山云H20实测DeepSeek开源的3FS

## 1 DeepSeek 3FS分布式存储概述
3FS 满足 AI 处理过程中的大部分场景：
1. Training data preprocessing数据集的预处理需求。
2. Dataset loading训练过程中的数据集读取需求 。
3. Checkpointing训练过程中，高并发 checkpoint 文件的写入。
4. KVCache for Inference为 KVCache 提供了比 DRAM 更加经济的替代方案，提供更低的成本和更大的容量（回答更smart）
5. Embedding vector search提供向量搜索功能的持久化。
系统架构
整体架构
![\[图片\]](https://i-blog.csdnimg.cn/direct/3c31bfb324734582b6138952ff0b3713.png)
### 核心组件
- Cluster Manager：用于管理整个集群的状态和 FO ，其它组件通过 Heartbeat 和 Cluster Manager 交互，Cluster Manager：本身通过多副本的选主机制来保证自身的可靠性，使用 ZK 或者 ETCD。
- Metadata Service：存储文件元数据，3FS 的元数据使用独立/外置的 FundationDB 集群来存储，FoundationDB 本身是一个完备的分布式 KV DB，Metadata 的数据可靠性由 FoundationDB 来保证。
- Storage Service：存储数据，数据同步协议是 CRAQ，Storage Service 自身来保障数据的高可用性。
- Client：对性能不敏感的应用使用 Fuse 客户端，对性能要求高的应用使用 native 客户端。
### 外部依赖
- ClickHouse：用于存储服务产生的 Metrics。
- FoundationDB：用于元数据服务存储文件元数据。
- Zookeeper/etcd：用于 ClusterManager 实现多副本选主。


## 2 构建编译环境

首先我们在火山云上申请一个ecs.hpcpni3ln.45xlarge实例作为编译环境使用. 注意在创建实例的时候,选择unbuntu 22.04

记得设置代理：

```bash
export https_proxy=http://172.17.0.1:1081
export http_proxy=http://172.17.0.1:1081
```

首先安装编译需要的package：

```bash
sudo apt install libicu70=70.1-2 apt install cmake libicu-dev libuv1-dev liblz4-dev liblzma-dev libdouble-conversion-dev libprocps-dev libdwarf-dev libunwind-dev \ libaio-dev libgflags-dev libgoogle-glog-dev libgtest-dev libgmock-dev clang-format-14 clang-14 clang-tidy-14 lld-14 \ libgoogle-perftools-dev google-perftools libssl-dev ccache gcc-12 g++-12 libboost-all-dev
```

然后安装libfuse，需要注意使用fuse3.16以上的版本：

```bash
wget https://github.com/libfuse/libfuse/releases/download/fuse-3.16.1/fuse-3.16.1.tar.gz
tar vzxf fuse-3.16.1.tar.gz
cd fuse-3.16.1/
mkdir build; cd build
apt install meson meson setup .. ninja ; ninja install
```

安装rust工具链：

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

安装foundation db：

```bash
wget https://github.com/apple/foundationdb/releases/download/7.3.63/foundationdb-clients_7.3.63-1_amd64.deb
wget https://github.com/apple/foundationdb/releases/download/7.3.63/foundationdb-server_7.3.63-1_amd64.deb
dpkg -i foundationdb-clients_7.3.63-1_amd64.deb
dpkg -i foundationdb-server_7.3.63-1_amd64.deb
vim /etc/foundationdb/foundationdb.conf
```

```bash
fdbcli --exec "configure new single memory"
```

## 3 编译3fs

按照如下方式下载和编译3fs：

```bash
git clone https://github.com/deepseek-ai/3fs
cd 3fs
git submodule update --init --recursive
./patches/apply.sh
cmake -S . -B build -DCMAKE_CXX_COMPILER=clang++-14 -DCMAKE_C_COMPILER=clang-14 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j 32
```

检查编译输出的binary：

```bash
root@3fs-1:~/3fs ls -lrt build/bin/
```

## 4 制作镜像

编译完成后对该台机器制作一个镜像用于后续部署。

## 5 配置文件修改

我们采用官方文档推荐的方式，在火山云上创建1个ecs.hpcpni3ln.45xlarge作为meta服务器和4个ecs.hpcpni3ln.45xlarge实例作为存储服务器。

修改配置文件中的max_sge：

```bash
cd ~/3fs/configs
sed -i 's/max_sge = 16/max_sge = 1/g' $(grep -rl max_sge)
```

另外由于3FS使用了mellanox网卡的ibdev2netdev，在执行3fs命令时会调用，所以，要保证OFED驱动是没问题的。

然后将meta对应的ip填入每个节点的/etc/hosts：

```bash
vim /etc/hosts
#添加 10.99.0.1 meta
```

每个节点的服务和相应的配置文件和官方建议相同，如下所示：

Service | Binary | Config files | NodeID | Node
---|---|---|---|---
monitor | monitor_collector_main | monitor_collector_main.toml | - | meta
admin_cli | admin_cli | admin_cli.toml
fdb.cluster | - | meta
storage1
storage2
storage3
storage4
storage5
mgmtd | mgmtd_main | mgmtd_main_launcher.toml<br>mgmtd_main.toml<br>mgmtd_main_app.toml | 1 | meta
meta | meta_main | meta_main_launcher.toml<br>meta_main.toml<br>meta_main_app.toml | 100 | meta
storage | storage_main | storage_main_launcher.toml<br>storage_main.toml<br>storage_main_app.toml | 10001~10005 | storage1<br>storage2<br>storage3<br>storage4<br>storage5
client | hf3fs_fuse_main | hf3fs_fuse_main_launcher.toml<br>hf3fs_fuse_main.toml | - | meta

## 6 安装ClickHouse和FoundationDB

由于复用了编译环境的镜像已经安装了FoundationDB，因此仅需在meta节点安装ClickHouse。

安装clickhouse，可以参考[ClickHouse官方文档](https://clickhouse.com/docs/install)：

```bash
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg
curl -fsSL 'https://packages.clickhouse.com/rpm/lts/repodata/repomd.xml.key' | sudo gpg --dearmor -o /usr/share/keyrings/clickhouse-keyring.gpg
ARCH=$(dpkg --print-architecture)
echo "deb [signed-by=/usr/share/keyrings/clickhouse-keyring.gpg arch=${ARCH}] https://packages.clickhouse.com/deb stable main" | sudo tee /etc/apt/sources.list.d/clickhouse.list
sudo apt-get update
sudo apt-get install -y clickhouse-server clickhouse-client
```

在安装的时候会要求输入密码，此时我们输入密码。

使用如下方式开启clickhouse服务：

```bash
sudo clickhouse start
```

然后使用安装时的密码验证登陆：

```bash
root@3fs-meta:~ clickhouse-client --password 'RDMA123!!'
```

然后退出，并采用如下命令创建Metric table：

```bash
clickhouse-client --password 'RDMA123!!' -n < ~/3fs/deploy/sql/3fs-monitor.sql
```

首次使用记得初始化集群数据：

```bash
fdbcli --exec "configure new single memory"
```

single memory 表示单节点内存模式（测试用）。生产环境需指定多节点和存储引擎（如 ssd）。

停止服务并清除数据目录：

```bash
sudo systemctl stop foundationdb
rm -rf /var/lib/foundationdb/data/*
```

重新初始化：

```bash
fdbcli --exec "configure new single memory"
sudo systemctl start foundationdb
```

## 7 配置监控服务

仅在meta节点配置安装monitor_collector服务。

```bash
mkdir -p /opt/3fs/{bin,etc}
mkdir -p /var/log/3fs
cp ~/3fs/build/bin/monitor_collector_main /opt/3fs/bin
cp ~/3fs/configs/monitor_collector_main.toml /opt/3fs/etc
```

修改monitor_collector_main.toml如下所示：

```toml
[server.monitor_collector.reporter.clickhouse]
db = '3fs'
host = '127.0.0.1'
passwd = 'passwd'
port = '9000'
user = 'default'
```

启动monitor_collector服务如下：

```bash
cp ~/3fs/deploy/systemd/monitor_collector_main.service /usr/lib/systemd/system
systemctl start monitor_collector_main
```

检查服务状态：

```bash
root@3fs-meta:/opt/3fs/etc systemctl status monitor_collector_main
```

## 8 配置Admin Client

在所有节点安装admin_cli：

```bash
mkdir -p /opt/3fs/{bin,etc}
rsync -avz meta:~/3fs/build/bin/admin_cli /opt/3fs/bin
rsync -avz meta:~/3fs/configs/admin_cli.toml /opt/3fs/etc
rsync -avz meta:/etc/foundationdb/fdb.cluster /opt/3fs/etc
```

更新admin_cli.toml：

```toml
cluster_id = "stage"
[fdb]
clusterFile = '/opt/3fs/etc/fdb.cluster'
```

admin_cli的使用帮助文档可以输入：

```bash
root@3fs-meta:/opt/3fs/etc /opt/3fs/bin/admin_cli -cfg /opt/3fs/etc/admin_cli.toml help bench
```

## 9 配置Mgmtd Service

mgmtd仅在meta节点安装。首先拷贝文件：

```bash
cp ~/3fs/build/bin/mgmtd_main /opt/3fs/bin
cp ~/3fs/configs/{mgmtd_main.toml,mgmtd_main_launcher.toml,mgmtd_main_app.toml} /opt/3fs/etc
```

修改配置文件，将mgmtd配置文件mgmtd_main_app.toml定义node_id =1：

```toml
node_id = 1
```

修改/opt/3fs/etc/mgmtd_main_launcher.toml中的cluster_id和clusterFile：

```toml
cluster_id = "stage"
[fdb]
clusterFile = '/opt/3fs/etc/fdb.cluster'
```

这里如果出现报错：可能是对应的IB devices不对，需要指定IB devices序列：

```toml
[ib_devices]
device_filter = ['mlx5_1','mlx5_2','mlx5_3','mlx5_4']
```

修改mgmtd_main.toml将remote ip修改为meta服务器地址：

```toml
[common.monitor.reporters.monitor_collector]
remote_ip = "10.99.0.1:10000"
```

配置完成后，初始化集群：

```bash
/opt/3fs/bin/admin_cli -cfg /opt/3fs/etc/admin_cli.toml "init-cluster --mgmtd /opt/3fs/etc/mgmtd_main.toml 1 1048576 16"
```

这步如果报错Encounter error: 6004(MgmtdClient::RoutingInfoNotReady)，可以看下IBManager是否启动：

```bash
systemctl start opensmd
```

如果报错以下内容，尝试重装IB驱动：

```bash
[2025-03-13T09:53:16.065019780+08:00 CliConn1:1115422 IBConnect.cc:418 ERROR] IBSocket [connect RDMA://10.99.48.9:8000] failed to connect, error RPC::ConnectFailed(2014) IBSocket failed to modify QP to RTR., timeout 5s
```

其中参数1代表chainTable ID, 1048576代表chunksize, 16代表file strip size。然后启动服务并验证：

```bash
cp ~/3fs/deploy/systemd/mgmtd_main.service /usr/lib/systemd/system
systemctl start mgmtd_main
```

检查节点：

```bash
root@3fs-meta:~ /opt/3fs/bin/admin_cli -cfg /opt/3fs/etc/admin_cli.toml --config.mgmtd_client.mgmtd_server_addresses '["RDMA://10.99.0.1:8000"]' "list-nodes"
```

## 10 配置Meta Service

该服务仅在meta服务器安装，拷贝文件如下所示：

```bash
cp ~/3fs/build/bin/meta_main /opt/3fs/bin
cp ~/3fs/configs/{meta_main_launcher.toml,meta_main.toml,meta_main_app.toml} /opt/3fs/etc
```

修改meta_main_app.toml中的node_id = 100。修改meta_main_launcher.toml中的 cluster_id, clusterFile：

```toml
cluster_id = "stage"
[mgmtd_client]
mgmtd_server_addresses = ["RDMA://10.99.0.1:8000"]
```

修改meta_main.toml如下：

```toml
[server.mgmtd_client]
mgmtd_server_addresses = ["RDMA://10.99.0.1:8000"]
[common.monitor.reporters.monitor_collector]
remote_ip = "10.99.0.1:10000"
[server.fdb]
clusterFile = '/opt/3fs/etc/fdb.cluster'
```

更新配置如下：

```bash
/opt/3fs/bin/admin_cli -cfg /opt/3fs/etc/admin_cli.toml --config.mgmtd_client.mgmtd_server_addresses '["RDMA://10.99.0.1:8000"]' "set-config --type META --file /opt/3fs/etc/meta_main.toml"
```

启动服务：

```bash
cp ~/3fs/deploy/systemd/meta_main.service /usr/lib/systemd/system
systemctl start meta_main
```

检查节点：

```bash
root@3fs-meta:~/opt/3fs/bin/admin_cli -cfg /opt/3fs/etc/admin_cli.toml --config.mgmtd_client.mgmtd_server-addresses '["RDMA://10.99.0.1:8000"]' "list-nodes"
```

## 11 配置Storage Service

在所有存储节点启用storage服务，由于我们每个节点只有8块盘，配置挂载如下：

```bash
mkdir -p /storage/data{0..7}
mkdir -p /var/log/3fs
for i in {0..7};do mkfs.xfs -L data${i} /dev/nvme${i}n1;mount -o noatime,nodiratime -L data${i} /storage/data${i};done
mkdir -p /storage/data{0..7}/3fs
```

增加aio请求的最大数：

```bash
sysctl -w fs.aio-max-nr=67108864
```

修改meta节点的原始配置文件~/3fs/configs/storage_main_launcher.toml中的clusterid和管理地址：

```toml
cluster_id = "stage"
[mgmtd_client]
mgmtd_server_addresses = ["RDMA://10.99.0.1:8000"]
```

修改~/3fs/configs/storage_main.toml中的IP地址和target path：

```toml
[server.mgmtd]
mgmtd_server_address = ["RDMA://10.99.0.1:8000"]
[common.monitor.reporters.monitor_collector]
remote_ip = "10.99.0.1:10000"
[server.targets]
target_paths = ["/storage/data0/3fs","/storage/data1/3fs","/storage/data2/3fs","/storage/data3/3fs","/storage/data4/3fs","/storage/data5/3fs","/storage/data6/3fs","/storage/data7/3fs"]
```

从meta节点拷贝执行文件和配置文件：

```bash
rsync -avz meta:~/3fs/build/bin/storage_main /opt/3fs/bin
rsync -avz meta:~/3fs/configs/{storage_main_launcher.toml,storage_main.toml,storage_main_app.toml} /opt/3fs/etc
```

每个存储节点修改/opt/3fs/etc/storage_main_app.toml中的node_id，五台机器分别为10001~10005。

指定IB devices序列：

```toml
[ib_devices]
device_filter = ['mlx5_1','mlx5_2','mlx5_3','mlx5_4']
```

然后每个存储节点更新：

```bash
/opt/3fs/bin/admin_cli -cfg /opt/3fs/etc/admin_cli.toml --config.mgmtd_client.mgmtd_server-addresses '["RDMA://10.99.0.1:8000"]' "set-config --type STORAGE --file /opt/3fs/etc/storage_main.toml"
```

最后启动并验证服务：

```bash
rsync -avz meta:~/3fs/deploy/systemd/storage_main.service /usr/lib/systemd/system
systemctl start storage_main
```

检查系统节点：

```bash
root@3fs-storage001:~/opt/3fs/bin/admin_cli -cfg /opt/3fs/etc/admin_cli.toml --config.mgmtd_client.mgmtd_server-addresses '["RDMA://10.99.0.1:8000"]' "list-nodes"
```

## 12 配置3FS

创建管理员：

```bash
root@3fs-meta:~/3fs/configs /opt/3fs/bin/admin_cli -cfg /opt/3fs/etc/admin_cli.toml --config.mgmtd_client.mgmtd_server-addresses '["RDMA://10.99.0.1:8000"]' "user-add --root --admin 0 root"
```

将token保存在/opt/3fs/etc/token.txt中。

然后创建chain Table，首先安装python相关的依赖：

```bash
pip3 install -r ~/3fs/deploy/data_placement/requirements.txt
```

然后执行data_placement计算命令：

```bash
root@3fs-meta python3 ~/3fs/deploy/data_placement/src/model/data_placement.py \ -ql -relax -type CR --num_nodes 5 --replication_factor 3 --min_targets_per_disk 6
```

然后执行产生chainTable：

```bash
python3 ~/3fs/deploy/data_placement/src/setup/gen_chain_table.py \ --chain_table_type CR --node_id_begin 10001 --node_id_end 10004 \ --num_disks_per_node 4 --num_targets_per_disk 6 \ --target_id_prefix 1 --chain_id_prefix 9 \ --incidence_matrix_path output/DataPlacementModel-v_5-b_10-r_6-k_3-λ_2-lb_1-ub_1/incidence_matrix.pickle
```

检查output目录是否产生了如下文件：

```bash
root@3fs-meta:/opt/3fs ls -lrt output
```

创建storage target：

```bash
/opt/3fs/bin/admin_cli --cfg /opt/3fs/etc/admin_cli.toml --config.mgmtd_client.mgmtd_server-addresses '["RDMA://10.99.0.1:8000"]' --config.user_info.token $(<"/opt/3fs/etc/token.txt") < output/create_target_cmd.txt
```

上传chains 和 chain table到mgmtd service：

```bash
/opt/3fs/bin/admin_cli --cfg /opt/3fs/etc/admin_cli.toml --config.mgmtd_client.mgmtd_server-addresses '["RDMA://10.99.0.1:8000"]' --config.user_info.token $(<"/opt/3fs/etc/token.txt") "upload-chains output/generated_chains.csv"
/opt/3fs/bin/admin_cli --cfg /opt/3fs/etc/admin_cli.toml --config.mgmtd_client.mgmtd_server-addresses '["RDMA://10.99.0.1:8000"]' --config.user_info.token $(<"/opt/3fs/etc/token.txt") "upload-chain-table --desc stage 1 output/generated_chain_table.csv"
```

检查是否上传成功：

```bash
/opt/3fs/bin/admin_cli -cfg /opt/3fs/etc/admin_cli.toml --config.mgmtd_client.mgmtd_server-addresses '["RDMA://10.99.0.1:8000"]' "list-chains"
/opt/3fs/bin/admin_cli -cfg /opt/3fs/etc/admin_cli.toml --config.mgmtd_client.mgmtd_server-addresses '["RDMA://10.99.0.1:8000"]' "list-chain-tables"
```

## 13 配置FUSE Client

在这个demo中我们采用在多个独立的节点部署FUSE Client的方式，首先拷贝文件，并创建mount点：

```bash
cp ~/3fs/build/bin/hf3fs_fuse_main /opt/3fs/bin
cp ~/3fs/configs/{hf3fs_fuse_main_launcher.toml,hf3fs_fuse_main.toml,hf3fs_fuse_main_app.toml} /opt/3fs/etc
mkdir -p /3fs/stage
```

修改/opt/3fs/etc/hf3fs_fuse_main_launcher.toml配置如下：

```toml
cluster_id = "stage"
mountpoint = '/3fs/stage'
token_file = '/opt/3fs/etc/token.txt'
[mgmtd_client]
mgmtd_server_addresses = ["RDMA://10.99.0.1:8000"]
[ib_devices]
device_filter = ['mlx5_1','mlx5_2','mlx5_3','mlx5_4']
```

修改/opt/3fs/etc/hf3fs_fuse_main.toml配置如下：

```toml
[mgmtd]
mgmtd_server_addresses = ["RDMA://10.99.48.9:8000"]
[common.monitor.reporters.monitor_collector]
remote_ip = "10.99.48.9:10000"
```

更新Fuse client配置到mgmtd service：

```bash
/opt/3fs/bin/admin_cli -cfg /opt/3fs/etc/admin_cli.toml --config.mgmtd_client.mgmtd_server-addresses '["RDMA://10.99.0.1:8000"]' "set-config --type FUSE --file /opt/3fs/etc/hf3fs_fuse_main.toml"
```

开启fuse client：

```bash
cp ~/3fs/deploy/systemd/hf3fs_fuse_main.service /usr/lib/systemd/system
systemctl start hf3fs_fuse_main
```

检查是否mount：

```bash
root@3fs-client:/opt/3fs mount | grep '/3fs/stage'
root@3fs-meta:/opt/3fs df -kh
```

## 14 性能测试

性能测试结果见：

## 15 其他

3FS还使用了clickhouse对运行数据进行统计分析，可以登陆meta节点的查询：

```bash
clickhouse-client --password 'RDMA123!!'
```

```sql
3fs-meta :) use 3fs
3fs-meta :) select * from distributions where metricName=='storage_client.request_bw' AND host=='3fs-fuse' limit 10
```

其它Metric可以通过如下命令查询：

```sql
3fs-meta :) select distinct metricName from distributions
```

## 参考资料

[1] ffrecord: [https://github.com/HFAiLab/ffrecord](https://github.com/HFAiLab/ffrecord)

[2] smallpond: [https://github.com/deepseek-ai/smallpond](https://github.com/deepseek-ai/smallpond)