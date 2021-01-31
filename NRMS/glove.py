import os
import requests
import math
from tqdm import tqdm
import logging
from contextlib import contextmanager
from tempfile import TemporaryDirectory
import zipfile
import bcolz
import numpy as np
import pickle

log = logging.getLogger(__name__)


def maybe_download(url, filename=None, work_directory=".", expected_bytes=None):
    """Download a file if it is not already downloaded.
       Reference: https://github.com/microsoft/recommenders/
    Args:
        filename (str): File name.
        work_directory (str): Working directory.
        url (str): URL of the file to download.
        expected_bytes (int): Expected file size in bytes.

    Returns:
        str: File path of the file downloaded.
    """
    if filename is None:
        filename = url.split("/")[-1]
    os.makedirs(work_directory, exist_ok=True)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):

        r = requests.get(url, stream=True)
        total_size = int(r.headers.get("content-length", 0))
        block_size = 1024
        num_iterables = math.ceil(total_size / block_size)

        with open(filepath, "wb") as file:
            for data in tqdm(
                    r.iter_content(block_size),
                    total=num_iterables,
                    unit="KB",
                    unit_scale=True,
            ):
                file.write(data)
    else:
        log.info("File {} already downloaded".format(filepath))
    if expected_bytes is not None:
        statinfo = os.stat(filepath)
        if statinfo.st_size != expected_bytes:
            os.remove(filepath)
            raise IOError("Failed to verify {}".format(filepath))

    return filepath


@contextmanager
def download_path(path=None):
    """Return a path to download data. If `path=None`, then it yields a temporal path that is eventually deleted,
    otherwise the real path of the input.
    Reference: https://github.com/microsoft/recommenders/
    Args:
        path (str): Path to download data.
    Returns:
        str: Real path where the data is stored.
    # Examples:
    #     >>> with download_path() as path:
    #     >>> ... maybe_download(url="http://example.com/file.zip", work_directory=path)
    """
    if path is None:
        tmp_dir = TemporaryDirectory()
        try:
            yield tmp_dir.name
        finally:
            tmp_dir.cleanup()
    else:
        path = os.path.realpath(path)
        yield path


def unzip_file(zip_src, dst_dir, clean_zip_file=True):
    """Unzip a file
    Args:
        zip_src (str): Zip file.
        dst_dir (str): Destination folder.
        clean_zip_file (bool): Whether or not to clean the zip file.
    """
    fz = zipfile.ZipFile(zip_src, "r")
    for file in fz.namelist():
        fz.extract(file, dst_dir)
    if clean_zip_file:
        os.remove(zip_src)


def download_and_extract_globe(dst_path):
    """
    Reference: https://github.com/microsoft/recommenders/
    :param dst_path: zip file download destination
    :return: glove file path
    """
    url = "http://nlp.stanford.edu/data/glove.6B.zip"
    file_path = maybe_download(url=url, work_directory=dst_path)
    unzip_file(file_path, dst_path, clean_zip_file=False)
    return dst_path


def generate_glove_vocab(glove_path, embedding_dim, max_vocab_size):
    """
    Partly refer to Reference: https://github.com/microsoft/recommenders/
    Use bcolz to save vocab embeddings into a persistent file with high compression rate
    :param glove_path: path to the GloVec file
    :param embedding_dim: embedding dimension
    :param max_vocab_size: vocab size
    :return:
    """
    embedding_path = f'{glove_path}/6B.'+str(embedding_dim)+'_words.pkl'
    index_path = f'{glove_path}/6B.'+str(embedding_dim)+'_idx.pkl'
    if (not os.path.exists(embedding_path)) or (not os.path.exists(index_path)):
        words = []
        word2idx = {}
        index = 1
        embedding_vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.' + str(embedding_dim) + '.dat', mode='w')
        embedding_vectors.append(np.zeros(embedding_dim))
        with open(f'{glove_path}/glove.6B.'+str(embedding_dim)+'d.txt', 'rb') as file:
            for line in tqdm(file):
                line = line.decode().split()
                word = line[0]
                words.append(word)
                word2idx[word] = index
                index += 1

                vec = np.array(line[1:]).astype(np.float)
                embedding_vectors.append(vec)
                if index > max_vocab_size:
                    break
        embedding_vectors = bcolz.carray(embedding_vectors[1:].reshape((-1, embedding_dim)),
                                         rootdir=f'{glove_path}/6B.' + str(embedding_dim) + '.dat',
                                         mode='w')
        embedding_vectors.flush()

        pickle.dump(words, open(f'{glove_path}/6B.'+str(embedding_dim)+'_words.pkl', 'wb'))
        pickle.dump(word2idx, open(f'{glove_path}/6B.'+str(embedding_dim)+'_idx.pkl', 'wb'))
    else:
        log.info("File {} and {} already downloaded".format(embedding_path, index_path))


if __name__ == "__main__":
    from config import hyperParams
    glove_pth = download_and_extract_globe(hyperParams["glove_path"])
    generate_glove_vocab(hyperParams["glove_path"],
                         hyperParams["model"]["embedding_size"],
                         hyperParams["max_vocab_size"])
