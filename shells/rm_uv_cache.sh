
#!/bin/bash

# 查看缓存目录
uv cache dir

# 清除没有用到的缓存数据
uv cache prune

# 彻底地删除cache
uv cache clean
