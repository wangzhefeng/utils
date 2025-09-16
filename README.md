<details><summary>目录</summary><p>

- [数据代码上传](#数据代码上传)
    - [文件压缩](#文件压缩)
        - [tar 文件](#tar-文件)
        - [gz 文件](#gz-文件)
        - [zip 文件](#zip-文件)
    - [文件上传](#文件上传)
    - [文件解压](#文件解压)
        - [tar 文件](#tar-文件-1)
        - [gz 文件](#gz-文件-1)
        - [zip 文件](#zip-文件-1)
- [服务启停](#服务启停)
    - [查看进程](#查看进程)
    - [杀进程](#杀进程)
    - [启动进程](#启动进程)
    - [监控进程](#监控进程)
- [Docker 环境](#docker-环境)
    - [启动 Docker](#启动-docker)
    - [进入 Docker](#进入-docker)
    - [停止 Docker](#停止-docker)
- [模型训练](#模型训练)
    - [单机单卡](#单机单卡)
    - [单机多卡](#单机多卡)
        - [数据并行](#数据并行)
        - [数据分布式并行](#数据分布式并行)
    - [多机多卡](#多机多卡)
- [Tmux](#tmux)
    - [新建](#新建)
    - [查看](#查看)
    - [进入](#进入)
- [Shell Script 相关](#shell-script-相关)
    - [选择行尾序列](#选择行尾序列)
- [参考资料](#参考资料)
</p></details><p></p>

# 数据代码上传

## 文件压缩

### tar 文件

```bash
$ tar -cvf  files.tar     files.txt  # 仅打包，不压缩
$ tar -zcvf files.tar.gz  files.txt  # 打包后，以 gzip 压缩
$ tar -jcvf files.tar.bz2 files.txt  # 打包后，以 bzip2 压缩
```

### gz 文件

```bash
$ gzip *           # 将所有文件压缩成 .gz 文件
$ gzip -l *        # 详细显示压缩文件的信息，并不解压

$ gzip -r log.tar  # 压缩一个 tar 备份文件，此时压缩文件的扩展名为.tar.gz
$ gzip -rv test/   # 递归的压缩目录
```

### zip 文件

```bash
$ zip -q -r file.zip /path/dir
```

## 文件上传

```bash
$ scp file.tar root@10.211.149.34:/root/workspace/projects/codes
```

## 文件解压

### tar 文件

```bash
$ tar -ztvf files.tar.gz          # 查阅上述 tar 包内有哪些文件
$ tar -zxvf file.tar.gz file.txt  # 只将 tar 内的部分文件解压出来
$ tar -zxvf files.tar.gz          # 将 tar 包解压缩
$ tar -xvf  files.tar             # 将 tar 包解压缩
```

### gz 文件

```bash
$ gzip -dv *       # 解压上例中的所有压缩文件，并列出详细的信息
$ gzip -dr test/   # 递归地解压目录
```

### zip 文件

```bash
$ unzip file.zip              # 解压 zip 文件

$ unzip file.zip /path/dir    # 在指定目录下解压缩
$ unzip -n file.zip -d /tmp/  # 在指定目录下解压缩
$ unzip -o file.zip -d /tmp/  # 在指定目录下解压缩，如果有相同文件存在则覆盖

$ unzip -v file.zip           # 查看压缩文件目录，但不解压
```

# 服务启停

## 查看进程

```bash
$ ps -aux | grep "python"
```

## 杀进程

```bash
$ kill -9 进程PID
```

## 启动进程

```bash
$ cd /codes/CES-load_prediction
$ nohup file.sh > logs.log 2>&1 &
```

## 监控进程

```bash
$ htop
```

# Docker 环境

## 启动 Docker

```bash
# 启动 docker
$ cd /workspace/projects
$ sh run.sh

# 判断 docker 是否启动
$ docker ps
```

## 进入 Docker

```bash
$ ssh -p 8848 root@127.0.0.1
$ DaMao2024!
```

## 停止 Docker

```bash
$ docker stop ces
```

# 模型训练

## 单机单卡

## 单机多卡

### 数据并行

> Data Parallelization

### 数据分布式并行

## 多机多卡

# Tmux

## 新建

```bash
$ tmux 
```

## 查看

```bash
$ tmux ls
```

## 进入

```bash
$ tmux a -t cmd_name
```

# Shell Script 相关

## 选择行尾序列

```bash
:set ff=unix
```




# 参考资料

* [Tmux 使用教程](https://www.ruanyifeng.com/blog/2019/10/tmux.html)
