import urllib.request
import os.path
import os
import gzip

url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(dataset_dir, "../mnist")

if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

def download(file_name):
    file_path = os.path.join(dataset_dir, file_name)
    if os.path.exists(file_path):
        print("Already download " + file_path)
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")

#unzip gzFile
def unzip_gz(file_name):
    # 获取文件的名称，去掉后缀名
    file_path = os.path.join(dataset_dir, file_name)
    target_file_path = file_path.replace(".gz", "")
    # 开始解压生成一个类
    print("Extracting files " + file_path + " ... ")
    g_file = gzip.GzipFile(file_path)
    # 读取数据部分(字节流）写入一个文件
    
    with open(target_file_path, "wb+") as f:
        f.write(g_file.read())
    g_file.close()
    print("Extract Done")


for file_name in key_file.values():
    download(file_name)

print("All mnist data download in " + dataset_dir)

for file_name in key_file.values():
    unzip_gz(file_name)

print("All mnist data extract in " + dataset_dir)