<details><summary>目录</summary><p>

- [uv 安装](#uv-安装)
- [uv 命令行工具](#uv-命令行工具)
- [uv 安装 python](#uv-安装-python)
    - [显式安装 Python](#显式安装-python)
    - [自动下载 Python](#自动下载-python)
    - [使用现有的 Python 版本](#使用现有的-python-版本)
    - [升级 Python 版本](#升级-python-版本)
- [uv 项目](#uv-项目)
    - [uv 创建新 Python 项目](#uv-创建新-python-项目)
    - [在工作目录中初始化一个项目](#在工作目录中初始化一个项目)
    - [uv 项目结构](#uv-项目结构)
    - [管理依赖项](#管理依赖项)
- [构建分发版本](#构建分发版本)
</p></details><p></p>


# uv 安装

* MacOS/Linux

```bash
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

* Windows

```bash
$ powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

# uv 命令行工具

* uv 安装的 Python 所在目录

```
C:\Users\Administrator\AppData\Roaming\uv\tools
```

* 使用 `uv tool install` 安装一个工具：

```bash
$ uv tool install ruff
$ ruff --version
```

* 使用 uvx (是 uv tool run 的别名) 在临时环境中运行一个工具：

```bash
$ uvx pycowsay 'hello world!'
```

# uv 安装 python

> uv 安装 Python 并允许快速切换版本。

* 查看可用的和已经安装的 Python 版本

```bash
$ uv python list
```

* uv 安装的 Python 所在目录

```
C:\Users\Administrator\AppData\Roaming\uv\python
```

## 显式安装 Python

* 安装最新版本的 Python

```bash
$ uv python install
```


* 安装多个 Python 版本：

```bash
$ uv python install 3.10 3.11 3.12
```

* 按需下载 Python 版本：

```bash
$ uv venv --python 3.12.0
```

* 在当前目录使用特定 Python 版本：

```bash
$ uv python pin 3.11
```

## 自动下载 Python

使用 uv 不需要显式安装 Python。默认情况下，当需要时，uv 会自动下载 Python 版本。

* 例如，如果未安装，以下命令将下载 Python 3.12：

```bash
$ uvx python@3.12 -c "print('hello world')"
```

## 使用现有的 Python 版本


## 升级 Python 版本

要升级到最新支持的补丁版本：

```bash
$ uv python upgrade 3.12
```

* 升级所有由 uv 管理的 Python 版本

```bash
$ uv python upgrade
```

# uv 项目

## uv 创建新 Python 项目

```bash
$ uv init hello-world
$ cd hello-world
$ uv add ruff
$ uv run ruff check
$ uv lock
$ uv sync
```

## 在工作目录中初始化一个项目

```bash
$ mkdir hello-world
$ cd hello-world
$ uv init
```

## uv 项目结构

完整的列表如下：

```
.
├── .venv
│   ├── bin
│   ├── lib
│   └── pyvenv.cfg
├── .python-version
├── README.md
├── main.py
├── pyproject.toml
└── uv.lock
```

* `pyproject.toml`

`pyproject.toml` 包含有关你的项目的元数据：

```
[project]
name = "hello-world"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
dependencies = []
```

你会使用这个文件来指定依赖项，以及项目的详细信息，例如其描述或许可证。
你可以手动编辑此文件，或使用 `uv add` 和 `uv remove` 等命令从终端管理你的项目。

* `.python-version`

`.python-version` 文件包含项目的默认 Python 版本。此文件告诉 uv 在创建项目的虚拟环境时使用哪个 Python 版本。

* `.venv`

`.venv` 文件夹包含你的项目的虚拟环境，这是一个与你的系统其他部分隔离的 Python 环境。这是 uv 安装你的项目依赖的地方。

* `uv.lock`

`uv.lock` 是一个跨平台的锁文件，其中包含你的项目依赖的精确信息。
与用于指定你的项目广泛需求的 `pyproject.toml` 不同，锁文件包含项目中安装的确切解析版本。
此文件应提交到版本控制中，以便在多台机器上实现一致和可重复的安装。

`uv.lock` 是一个人类可读的 TOML 文件，但由 uv 管理，不应手动编辑。

## 管理依赖项

可以使用 `uv add` 命令将依赖项添加到您的 `pyproject.toml` 中。这将同时更新锁文件和项目环境：

```bash
$ uv add requests
```

还可以指定版本约束或替代来源：

```bash
$ # Specify a version constraint
$ uv add 'requests=2.31.0'
$ # Add a git dependency
$ uv add git+https://github.com/psf/requests
```

如果你从 `requirements.txt` 文件迁移，可以使用 `uv add` 配合 `-r` 标志来添加文件中的所有依赖：

```bash
$ # Add all dependencies from `requirements.txt`.
uv add -r requirements.txt -c constraints.txt
```

要移除一个包，可以使用 `uv remove`：

```bash
$ uv remove requests
```

要升级一个包，运行 `uv lock` 配合 `--upgrade-package` 标志：

```bash
$ uv lock --upgrade-package requests
```

`--upgrade-package` 标志将尝试将指定包更新到最新兼容版本，同时保持 lockfile 的其他部分不变。

# 构建分发版本

`uv build` 可用于为您的项目构建源分发版本和二进制分发版本（wheel）。

默认情况下，`uv build` 将在当前目录中构建项目，并将构建的工件放置在 `dist/` 子目录中：

```bash
$ uv build
$ ls dist/
hello-world-0.1.0-py3-none-any.whl
hello-world-0.1.0.tar.gz
```

